"""
Progressive pruning pipeline (single resident process).

Generate i2v candidates 33 px-frames per chunk and progressively prune with QIP:
  64 -> 48 -> 12 -> 4   (configurable via --prune_schedule)
At each checkpoint (after each 33-frame chunk): generate next chunk for SURVIVORS only
(true-incremental via HeliosPipeline resume_state) -> SAM2 propagation -> optical flow
-> QIP selection (motion-diversity + CLIP unary). All models stay resident; the persistent
inductor cache avoids recompiles. Reports load-excluded compute time per stage.

Generation is batch=1 sequential per seed (the 14B model OOMs at large batch), keeping each
seed's autoregressive state in `states[seed]` and resuming it chunk-by-chunk so earlier
chunks are NOT re-generated (that is what saves generation compute).

Usage (see scripts/inference/progressive_pruning_pbs.sh):
  python tools/progressive_pruning.py --num_seeds 64 --prune_schedule 48 12 4 \
      --image_path output_helios/fish_i2v_first_frame.png --prompt "<fish prompt>" \
      --output_dir output_helios/progressive_pruning
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# Persistent inductor cache (must precede `import torch`); shared with the SAM2 path.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_ROOT / ".torchinductor_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Repo root on path for `helios.*`; tools/ for select_diverse_videos.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from diffusers.models import AutoencoderKLWan  # noqa: E402
from diffusers.utils import export_to_video, load_image  # noqa: E402
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large  # noqa: E402

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline  # noqa: E402
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler  # noqa: E402
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel  # noqa: E402
from helios.modules.helios_kernels import (  # noqa: E402
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)

# Pure reusable selection functions (safe to import; no bad side effects).
from select_diverse_videos import (  # noqa: E402
    compute_flow_raft,
    compute_horizon_displacement,
    compute_unary_clip,
    pairwise_distance,
    qip_greedy,
)
from sam2_common import best_box, detect_boxes_with_gdino  # noqa: E402

NEG = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
       "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
       "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
       "misshapen limbs, fused fingers, still picture, messy background, three legs, "
       "many people in the background, walking backwards")


def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _reduce_or(m: torch.Tensor, h: int) -> torch.Tensor:
    """[T,H,W] bool -> [T-h,H,W] via OR over each (h+1)-frame window."""
    T = m.shape[0]
    out = torch.empty(T - h, *m.shape[1:], device=m.device, dtype=torch.bool)
    for t in range(T - h):
        out[t] = m[t:t + h + 1].any(dim=0)
    return out


def _resize_mask(m: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if m.shape[-2:] == (H, W):
        return m
    mf = F.interpolate(m.float().unsqueeze(1), size=(H, W), mode="nearest")
    return mf.squeeze(1).bool()


def frames_to_uint8(video_np) -> torch.Tensor:
    """postprocess_video(output_type='np') -> uint8 [T,3,H,W] (batch row 0)."""
    arr = np.asarray(video_np)
    if arr.ndim == 5:        # [B,T,H,W,C]
        arr = arr[0]
    u8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)  # [T,H,W,C]
    return torch.from_numpy(u8).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]


def write_mp4(frames_u8: torch.Tensor, path: str, fps: int = 24):
    # export_to_video treats np.ndarray frames as float in [0,1] and does (frame*255);
    # passing uint8 [0,255] would overflow (uint8 * 255 -> ~255-v, a color INVERSION).
    # Hand it float32 [0,1] so the *255 reconstructs the correct uint8 RGB.
    frames = [frames_u8[i].permute(1, 2, 0).numpy().astype(np.float32) / 255.0
              for i in range(frames_u8.shape[0])]
    export_to_video(frames, path, fps=fps)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# ---- SAM2 vos_optimized compile patch (mode=default; copied from track_videos_native) ----
def patch_vos_compile_mode():
    import sam2.sam2_video_predictor as _svp

    def _patched(self):
        self.memory_encoder.forward = torch.compile(self.memory_encoder.forward, mode="default", fullgraph=True, dynamic=False)
        self.memory_attention.forward = torch.compile(self.memory_attention.forward, mode="default", fullgraph=True, dynamic=True)
        self.sam_prompt_encoder.forward = torch.compile(self.sam_prompt_encoder.forward, mode="default", fullgraph=True, dynamic=False)
        self.sam_mask_decoder.forward = torch.compile(self.sam_mask_decoder.forward, mode="default", fullgraph=True, dynamic=False)

    _svp.SAM2VideoPredictorVOS._compile_all_components = _patched


@torch.inference_mode()
def sam2_track(predictor, mp4_path, bbox, device):
    """Re-track a (growing) clip from an mp4. Returns bool mask [T,H,W].
    (Legacy non-incremental path; kept for the warmup / fallback.)"""
    # bf16 autocast scoped to SAM2 only (do NOT enter globally; would hit Helios' fp32 VAE).
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(mp4_path))
        T, H, W = state["num_frames"], state["video_height"], state["video_width"]
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        box=np.array(bbox, dtype=np.float32))
        masks = torch.zeros((T, H, W), dtype=torch.bool)
        for fidx, _ids, logits in predictor.propagate_in_video(state):
            masks[fidx] = (logits[0, 0] > 0.0).cpu()
        predictor.reset_state(state)
    return masks


# ---- INCREMENTAL SAM2: keep one inference_state per seed alive; only encode + propagate the
# NEW frames each checkpoint (old frames' memory features persist in the state's output_dict,
# so the hiera image encoder never re-runs on already-tracked frames). Matches the mp4 loader's
# preprocessing (resize to image_size^2, /255, normalize) so appended frames are consistent. ----
_SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_SAM2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _sam2_preprocess(frames_u8, image_size, device):
    """uint8 [n,3,H,W] RGB -> normalized float [n,3,S,S] on `device` (mirror load_video_frames)."""
    x = frames_u8.float() / 255.0
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    x = (x - _SAM2_MEAN.to(x.device)) / _SAM2_STD.to(x.device)
    return x.to(device)


@torch.inference_mode()
def sam2_init(predictor, frames_u8, bbox, tmp_mp4):
    """Start a persistent SAM2 state on the first chunk: init from an mp4 (offloaded to CPU),
    seed the box at frame 0, propagate the whole chunk. Returns (state, masks[T,H,W] bool)."""
    write_mp4(frames_u8, tmp_mp4)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(tmp_mp4),
                                     offload_video_to_cpu=True, offload_state_to_cpu=False)
        T, H, W = state["num_frames"], state["video_height"], state["video_width"]
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        box=np.array(bbox, dtype=np.float32))
        masks = torch.zeros((T, H, W), dtype=torch.bool)
        for fidx, _ids, logits in predictor.propagate_in_video(state):
            masks[fidx] = (logits[0, 0] > 0.0).cpu()
    return state, masks


@torch.inference_mode()
def sam2_extend(predictor, state, new_frames_u8):
    """Append new frames to a persistent state and propagate ONLY them (re-using the memory
    bank of already-tracked frames). Returns masks[n_new,H,W] bool for the new frames."""
    old_T = state["num_frames"]
    new_imgs = _sam2_preprocess(new_frames_u8, predictor.image_size, state["images"].device)
    state["images"] = torch.cat([state["images"], new_imgs], dim=0)
    n_new = new_imgs.shape[0]
    state["num_frames"] = old_T + n_new
    H, W = state["video_height"], state["video_width"]
    masks = torch.zeros((n_new, H, W), dtype=torch.bool)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for fidx, _ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=old_T, max_frame_num_to_track=n_new):
            if fidx >= old_T:
                masks[fidx - old_T] = (logits[0, 0] > 0.0).cpu()
    return masks


def _clip_tensor(out):
    """Some transformers versions return a model-output object from get_*_features."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v[:, 0] if attr == "last_hidden_state" else v
    raise TypeError(f"Unexpected CLIP output type: {type(out)}")


@torch.inference_mode()
def clip_embed(frames_u8, m_clip, text_emb, mean, std, device, batch=128):
    """uint8 [n,3,H,W] -> L2-normed CLIP image embeddings [n,D]. Each frame embedded once."""
    embs = []
    for i in range(0, frames_u8.shape[0], batch):
        sel = frames_u8[i:i + batch].to(device).float() / 255.0
        sel = F.interpolate(sel, size=(224, 224), mode="bilinear", align_corners=False)
        sel = (sel - mean) / std
        e = _clip_tensor(m_clip.get_image_features(pixel_values=sel))
        embs.append(F.normalize(e, dim=-1))
    return torch.cat(embs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_seeds", type=int, default=64)
    ap.add_argument("--prune_schedule", type=int, nargs="+", default=[48, 12, 4],
                    help="K kept after each chunk (len = number of chunks/checkpoints).")
    ap.add_argument("--model", default="BestWishYsh/Helios-Distilled")
    ap.add_argument("--image_path", required=True, help="i2v conditioning first frame (png).")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--pyramid_steps", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--lambda_score", type=float, default=2.0)
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip_frames", type=int, default=8)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam2_ckpt", default=str(_ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    ap.add_argument("--gdino_text", default="a fish.")
    ap.add_argument("--enable_compile", action="store_true")
    ap.add_argument("--warmup", action="store_true",
                    help="With --enable_compile, run an untimed dummy pass first so compile "
                         "is excluded from the per-stage timers.")
    ap.add_argument("--tmp_dir", default="/tmp/pp_work")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    K = args.prune_schedule
    num_chunks = len(K)

    # ---------------- load Helios (CWD = repo) ----------------
    print("[load] Helios pipeline...")
    dtype = torch.bfloat16
    transformer = HeliosTransformer3DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception:
        transformer.set_attention_backend("flash_hub")
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(args.model, transformer=transformer, vae=vae,
                                          scheduler=scheduler, torch_dtype=dtype).to(device)
    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    # ---------------- RAFT ----------------
    print("[load] RAFT-large...")
    raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).to(device).eval()

    # ---------------- conditioning image + GDINO bbox ----------------
    image = load_image(args.image_path).resize((args.width, args.height))
    bbox = best_box(*detect_boxes_with_gdino(image, args.gdino_text,
                                             "IDEA-Research/grounding-dino-tiny", device)[:2])

    # ---------------- SAM2 (import with ./sam2 namespace shadow removed) ----------------
    # `import sam2` must resolve to the editable-installed package (_ROOT/sam2/sam2), NOT the
    # repo dir _ROOT/sam2 (a namespace pkg containing sam2/). Drop _ROOT from sys.path + cd /tmp
    # for the import, then restore both (helios needs _ROOT on path for any lazy imports).
    print("[load] SAM2 predictor...")
    _saved_sys_path, _saved_cwd = list(sys.path), os.getcwd()
    os.chdir("/tmp")
    sys.path[:] = [p for p in sys.path if p not in ("", str(_ROOT))]
    patch_vos_compile_mode()
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device, vos_optimized=True)
    sys.path[:] = _saved_sys_path
    os.chdir(_saved_cwd)

    # ---------------- CLIP (resident; embed each frame ONCE and cache) ----------------
    print("[load] CLIP...")
    from transformers import CLIPModel, CLIPProcessor  # noqa: E402
    m_clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    _clip_prep = CLIPProcessor.from_pretrained(args.clip_model)
    _txt = _clip_prep.tokenizer(args.prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        text_emb = F.normalize(_clip_tensor(m_clip.get_text_features(**_txt)), dim=-1)  # [1, D]
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    # ---------------- WARMUP (untimed): trigger all torch.compile so the timed loop is
    # fully warm and compile time is EXCLUDED from the per-stage measurements. Runs the
    # whole hook pipeline (gen all chunks + SAM2 + flow + CLIP) once on a throwaway seed. ----
    if args.enable_compile and args.warmup:
        print("[warmup] compiling via 1 dummy seed (untimed)...")
        t_w = time.perf_counter()
        wg = torch.Generator(device=device).manual_seed(10**6)
        wst, wframes = None, None
        for ci in range(num_chunks):
            cw = dict(prompt=args.prompt, negative_prompt=None, height=args.height, width=args.width,
                      num_frames=33 * num_chunks, guidance_scale=1.0, is_enable_stage2=True,
                      pyramid_num_inference_steps_list=args.pyramid_steps, is_amplify_first_chunk=True,
                      num_latent_frames_per_chunk=9, history_sizes=[16, 2, 1], keep_first_frame=True,
                      generator=wg, output_type="np", return_dict=False, num_chunks=1, return_state=True)
            if ci == 0:
                vnp, wst = pipe(image=image, image_noise_sigma_min=0.111, image_noise_sigma_max=0.135, **cw)
            else:
                vnp, wst = pipe(image=None, resume_state=wst, **cw)
            wframes = frames_to_uint8(vnp)
        # warm SAM2 (both init-propagate AND incremental-extend graphs) + RAFT + CLIP
        wstate, _ = sam2_init(predictor, wframes[:33], bbox, str(tmp / "warmup.mp4"))
        if wframes.shape[0] > 33:
            _ = sam2_extend(predictor, wstate, wframes[33:65])
        predictor.reset_state(wstate)
        _ = compute_flow_raft(raft, wframes[:34], device, batch=8, num_flow_updates=8, compute_dtype=torch.bfloat16)
        _ = clip_embed(wframes[:8], m_clip, text_emb, CLIP_MEAN, CLIP_STD, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[warmup] done in {time.perf_counter() - t_w:.1f}s (excluded from measurements)")

    # ============== progressive loop (batch=1 sequential) ==============
    seeds = list(range(args.num_seeds))
    gens = {s: torch.Generator(device=device).manual_seed(s) for s in seeds}
    states = {s: None for s in seeds}          # resume_state per seed
    frames_u8 = {s: None for s in seeds}        # cumulative uint8 [T,3,H,W]
    prev_count = {s: 0 for s in seeds}          # frames before current chunk (flow seam)
    flow_acc = {s: None for s in seeds}         # [2,T-1,H,W] cpu
    sam2_states = {s: None for s in seeds}      # persistent SAM2 inference_state per seed
    sam2_masks = {s: None for s in seeds}       # cumulative bool mask [T,H,W] cpu
    sam2_tracked = {s: 0 for s in seeds}        # frames already tracked by SAM2
    clip_emb = {s: None for s in seeds}         # cached CLIP frame embeddings [n,D]
    clip_count = {s: 0 for s in seeds}          # frames already embedded
    alive = list(seeds)
    timings = []
    survivors = {"initial": list(alive)}

    for ci in range(num_chunks):
        print(f"\n===== checkpoint {ci}: chunk gen for {len(alive)} survivors =====")
        # ---- 1. GENERATION (next chunk for survivors only) ----
        t0 = sync()
        for s in alive:
            common = dict(prompt=args.prompt, negative_prompt=None, height=args.height, width=args.width,
                          num_frames=33 * num_chunks, guidance_scale=1.0, is_enable_stage2=True,
                          pyramid_num_inference_steps_list=args.pyramid_steps, is_amplify_first_chunk=True,
                          num_latent_frames_per_chunk=9, history_sizes=[16, 2, 1], keep_first_frame=True,
                          generator=gens[s], output_type="np", return_dict=False,
                          num_chunks=1, return_state=True)
            if ci == 0:
                video_np, st = pipe(image=image, image_noise_sigma_min=0.111, image_noise_sigma_max=0.135, **common)
            else:
                video_np, st = pipe(image=None, resume_state=states[s], **common)
            states[s] = st
            frames_u8[s] = frames_to_uint8(video_np)
        t_gen = sync() - t0

        # ---- 2. SAM2 propagation (INCREMENTAL: only the new frames; memory persists) ----
        t0 = sync()
        for s in alive:
            cur = frames_u8[s].shape[0]
            if sam2_states[s] is None:                       # first chunk: init + track all
                sam2_states[s], sam2_masks[s] = sam2_init(
                    predictor, frames_u8[s], bbox, str(tmp / f"sam2_init_seed{s}.mp4"))
            else:                                            # later chunks: track only new frames
                nm = sam2_extend(predictor, sam2_states[s], frames_u8[s][sam2_tracked[s]:cur])
                sam2_masks[s] = torch.cat([sam2_masks[s], nm], dim=0)
            sam2_tracked[s] = cur
        masks = sam2_masks
        t_sam2 = sync() - t0

        # ---- 3. FLOW (incremental: only the new chunk + 1 boundary frame) ----
        t0 = sync()
        for s in alive:
            cur = frames_u8[s].shape[0]
            start = max(0, prev_count[s] - 1)            # include 1 boundary frame for ci>0
            window = frames_u8[s][start:cur]             # uint8 [win,3,H,W]
            new_flow = compute_flow_raft(raft, window, device, batch=8,
                                         num_flow_updates=8, compute_dtype=torch.bfloat16)  # [2,win-1,H,W]
            flow_acc[s] = new_flow if flow_acc[s] is None else torch.cat([flow_acc[s], new_flow], dim=1)
            assert flow_acc[s].shape[1] == cur - 1, f"flow seam mismatch {flow_acc[s].shape[1]} vs {cur-1}"
            prev_count[s] = cur
        t_flow = sync() - t0

        # ---- 4. SELECTION (motion-diversity + CLIP unary, QIP) ----
        t0 = sync()
        Hf, Wf = flow_acc[alive[0]].shape[-2:]
        motion, combined = [], []
        for s in alive:
            f = flow_acc[s].to(device)
            h = min(args.horizon, f.shape[1])
            disp, vis = compute_horizon_displacement(f, h)
            mred = _resize_mask(_reduce_or(masks[s].to(device), h), Hf, Wf)
            motion.append(disp)
            combined.append((mred & vis))
        N = len(alive)
        D = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                d = pairwise_distance(motion[i], motion[j], combined[i] & combined[j])
                D[i, j] = D[j, i] = d
        # CLIP unary (INCREMENTAL: embed only new frames, reuse cached embeddings; no mp4 decode,
        # no model reload). unary = mean text-image cosine over all frames embedded so far.
        for s in alive:
            cur = frames_u8[s].shape[0]
            if clip_count[s] < cur:
                new = clip_embed(frames_u8[s][clip_count[s]:cur], m_clip, text_emb,
                                 CLIP_MEAN, CLIP_STD, device)
                clip_emb[s] = new if clip_emb[s] is None else torch.cat([clip_emb[s], new], dim=0)
                clip_count[s] = cur
        u = np.array([float((clip_emb[s] @ text_emb.T).mean().item()) for s in alive],
                     dtype=np.float32)
        keep_local = qip_greedy(_minmax(u), _minmax(D), K[ci], args.lambda_score)
        kept = [alive[i] for i in sorted(keep_local)]
        # release state for pruned seeds (free SAM2 memory bank + cached frames/embeddings)
        for s in set(alive) - set(kept):
            if sam2_states[s] is not None:
                predictor.reset_state(sam2_states[s])
            sam2_states[s] = sam2_masks[s] = clip_emb[s] = states[s] = frames_u8[s] = flow_acc[s] = None
        alive = kept
        t_sel = sync() - t0

        timings.append({"checkpoint": ci, "frames": int(frames_u8[alive[0]].shape[0]),
                        "n_in": N, "n_out": len(alive),
                        "gen": round(t_gen, 2), "sam2": round(t_sam2, 2),
                        "flow": round(t_flow, 2), "select": round(t_sel, 2)})
        survivors[f"after_chunk{ci}"] = list(alive)
        print(f"[TIME] ckpt{ci}: gen={t_gen:.2f}s sam2={t_sam2:.2f}s flow={t_flow:.2f}s "
              f"select={t_sel:.2f}s  ({N}->{len(alive)})")

    # ---------------- outputs ----------------
    for rank, s in enumerate(alive):
        write_mp4(frames_u8[s], str(out_dir / f"rank{rank}_seed{s}.mp4"))
    (out_dir / "survivors.json").write_text(json.dumps(survivors, indent=2))
    total = {k: round(sum(t[k] for t in timings), 2) for k in ("gen", "sam2", "flow", "select")}
    total["all"] = round(sum(total.values()), 2)
    (out_dir / "timings.json").write_text(json.dumps({"per_checkpoint": timings, "total": total}, indent=2))
    print("\n==== TIMING TOTAL (load-excluded) ====")
    print(json.dumps({"per_checkpoint": timings, "total": total}, indent=2))
    print(f"final survivors (seeds): {alive}")


if __name__ == "__main__":
    main()
