"""
Track an object across all frames of every mp4 in --video_dir using
**native** SAM2 (facebookresearch/sam2 submodule) instead of the HuggingFace
transformers wrapper.

Why a separate script:
  The HF Sam2VideoModel wrapper currently lacks the optimizations the SAM2
  benchmark.py uses (torch.compile of memory_encoder/memory_attention/prompt
  encoder/mask decoder + image-encoder compile + torch.autocast + TF32).
  Measured on the Helios 97x384x640 sample with SAM2.1-hiera-tiny:
      HF + bf16 (track_videos_with_sam2.py):   ~24.6 FPS
      Native, no compile:                       ~46.0 FPS
      Native + vos_optimized (mode=default):    ~61.1 FPS   ← this script
      Spec (mode=max-autotune, torch 2.5.1):    91.2 FPS

GroundingDINO bbox seeding is identical to the HF script.

Limitation: the SAM2 repository check at sam2/sam2/build_sam.py:20-32 errors
out if Python's CWD or sys.path[0] contains a `sam2/` directory (because the
HF Helios repo includes the sam2 submodule at ./sam2). This script `cd`s to
/tmp at startup so the installed `sam2` package is found instead of the
submodule shadowing it.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from video_reader import PyVideoReader

# Work around sam2's parent-dir shadow check (see module docstring).
# Doing this before `import sam2` is essential.
_HELIOS_ROOT = Path(__file__).resolve().parent.parent
os.chdir("/tmp")
# Drop "" (Helios cwd at script start) from sys.path[0] if present.
if sys.path and (sys.path[0] == "" or Path(sys.path[0]).resolve() == _HELIOS_ROOT):
    sys.path = sys.path[1:]

from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor  # noqa: E402

SEED_RE = re.compile(r"seed(\d+)_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--reference_image", type=str, default=None,
                   help="Image for GroundingDINO bbox detection. Defaults to first frame of first video.")
    p.add_argument("--text", type=str, default="a cat.",
                   help="GroundingDINO text prompt.")
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Skip GroundingDINO and use this bbox (video pixel coords).")
    p.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml",
                   help="SAM2 hydra config (relative to native sam2 package).")
    p.add_argument("--sam2_ckpt", type=str,
                   default=str(_HELIOS_ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt"),
                   help="Path to SAM2 .pt checkpoint (download with sam2/checkpoints/download_ckpts.sh).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable vos_optimized (skip torch.compile). Saves ~20s warmup but loses ~30% throughput.")
    p.add_argument("--video_glob", type=str, default="seed*.mp4")
    p.add_argument("--box_threshold", type=float, default=0.35)
    p.add_argument("--text_threshold", type=float, default=0.25)
    return p.parse_args()


def find_videos(video_dir: Path, glob_pat: str):
    out = []
    for p in sorted(video_dir.glob(glob_pat)):
        m = SEED_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_first_frame_pil(path: Path) -> Image.Image:
    vr = PyVideoReader(str(path), threads=0)
    buf = vr.decode()  # [T, H, W, 3] uint8
    del vr
    return Image.fromarray(buf[0])


def detect_bbox_with_gdino(image: Image.Image, text: str, gdino_model: str,
                           device: str, box_thr: float, text_thr: float) -> list[float]:
    processor = GroundingDinoProcessor.from_pretrained(gdino_model)
    model = GroundingDinoForObjectDetection.from_pretrained(gdino_model).to(device)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO found no boxes for prompt {text!r}")
    best = int(np.argmax(scores))
    bbox = boxes[best].tolist()
    print(f"GroundingDINO bbox: {bbox} (score={scores[best]:.3f})")
    del model, processor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return bbox


def patch_vos_compile_mode():
    """Force mode='default' for compile (torch 2.10 max-autotune triggers a known
    CUDAGraph output-overwrite error on SAM2's memory_attention)."""
    import sam2.sam2_video_predictor as _svp

    def _patched(self):
        print("[track_native] vos_optimized compile (mode=default)...")
        self.memory_encoder.forward = torch.compile(
            self.memory_encoder.forward, mode="default", fullgraph=True, dynamic=False)
        self.memory_attention.forward = torch.compile(
            self.memory_attention.forward, mode="default", fullgraph=True, dynamic=True)
        self.sam_prompt_encoder.forward = torch.compile(
            self.sam_prompt_encoder.forward, mode="default", fullgraph=True, dynamic=False)
        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward, mode="default", fullgraph=True, dynamic=False)

    _svp.SAM2VideoPredictorVOS._compile_all_components = _patched


@torch.inference_mode()
def track_one_video(predictor, video_path: Path, bbox_xyxy: list[float], device: str) -> torch.Tensor:
    """Run SAM2 on a single mp4. Returns bool tensor [T, H, W] (True = foreground)."""
    state = predictor.init_state(video_path=str(video_path))
    T = state["num_frames"]
    H, W = state["video_height"], state["video_width"]

    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        box=np.array(bbox_xyxy, dtype=np.float32),
    )

    masks = torch.zeros((T, H, W), dtype=torch.bool)
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        # mask_logits: [num_objs=1, 1, H, W] at video resolution
        m = (mask_logits[0, 0] > 0.0).to(torch.bool).cpu()
        masks[frame_idx] = m

    predictor.reset_state(state)
    return masks


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = _HELIOS_ROOT / video_dir
    videos = find_videos(video_dir, args.video_glob)
    if not videos:
        raise SystemExit(f"No videos in {video_dir}")
    print(f"Found {len(videos)} videos in {video_dir}")

    # Resolve seed bbox on the first frame of the first video (or user-provided)
    if args.bbox is not None:
        bbox = list(args.bbox)
        print(f"Using user-provided bbox: {bbox}")
    else:
        if args.reference_image:
            ref_img = Image.open(args.reference_image).convert("RGB")
        else:
            ref_img = load_first_frame_pil(videos[0][1])
        sample_h, sample_w = ref_img.size[1], ref_img.size[0]
        bbox = detect_bbox_with_gdino(
            ref_img, args.text, args.gdino_model, args.device,
            args.box_threshold, args.text_threshold,
        )

    # SAM2.1 spec config: bf16 autocast + TF32
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not args.no_compile:
        patch_vos_compile_mode()

    from sam2.build_sam import build_sam2_video_predictor
    print(f"Building SAM2 predictor (cfg={args.sam2_cfg}, ckpt={args.sam2_ckpt}, "
          f"vos_optimized={not args.no_compile})...")
    ckpt_path = args.sam2_ckpt
    if not Path(ckpt_path).is_absolute():
        ckpt_path = str(_HELIOS_ROOT / ckpt_path)
    if not Path(ckpt_path).exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download with: cd {_HELIOS_ROOT}/sam2/checkpoints && bash download_ckpts.sh"
        )
    predictor = build_sam2_video_predictor(
        args.sam2_cfg, ckpt_path, device=args.device,
        vos_optimized=not args.no_compile,
    )

    summary = {
        "bbox": bbox,
        "sam2_cfg": args.sam2_cfg,
        "sam2_ckpt": ckpt_path,
        "vos_optimized": not args.no_compile,
        "videos": {},
    }
    for seed, path in tqdm(videos, desc="track"):
        out_path = out_dir / f"seed{seed}.pt"
        if out_path.exists():
            mask = torch.load(out_path, map_location="cpu", weights_only=True)
        else:
            mask = track_one_video(predictor, path, bbox, args.device)
            torch.save(mask, out_path)
        fg_frac = mask.float().mean().item()
        summary["videos"][seed] = {"fg_frac_mean": fg_frac, "T": int(mask.shape[0])}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
