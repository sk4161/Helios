"""
Select the K most motion-diverse videos from a directory.

Motion representation: `--horizon N` (default 1)
  - N=1: instantaneous frame-to-frame optical flow (pixel-unit velocity field).
  - N>1: sliding-window cumulative displacement over the next N frames.
         Output shape [2, T-N, H, W]. Bounded drift (~σ·√N), preserves
         multi-frame motion patterns, no off-screen homogenization.
  - N=T-1: equivalent to a full-video trajectory from frame 0. NOTE: empirically
         worse on Helios output — RAFT-integration drift dominates the signal
         (see feedback_diversity_instantaneous_over_cumulative.md). Only use
         when an accurate long-range tracker (DOT) is available.

Visibility mask: pixels whose integrated position leaves [0, W) × [0, H) at any
step within the window are excluded from the distance. This replaces the
clamp(-1, 1) trick from MotionModes (which homogenized off-screen pixels into
the boundary value) with a principled exclusion.

Distance kernel: magnitude + angle (motion_losses_ours.py:991-1043). Works on
raw pixel displacements; the `flow*2-1` rescale and clamp from MotionModes are
dropped — those existed because MotionModes' "flow" was a [0, 1]-encoded image
decoded from a VAE latent (see train_ctrl_flow_gen_dot.py:625-646), not a raw
displacement field.

Selection: Farthest Point Sampling (FPS).
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm
from video_reader import PyVideoReader

# Allow importing memfof from the bundled package without installing it
_MEMFOF_PATH = Path(__file__).resolve().parent.parent / "memfof"
if _MEMFOF_PATH.exists() and str(_MEMFOF_PATH) not in sys.path:
    sys.path.insert(0, str(_MEMFOF_PATH))


SEED_RE = re.compile(r"seed(\d+)_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--mask_path", type=str, default=None,
                   help="Static 2D mask PNG (used as a fallback / shared mask). "
                        "If --mask_dir is also given, this is only used as a sanity reference.")
    p.add_argument("--mask_dir", type=str, default=None,
                   help="Directory of per-video time-varying masks `seed{N}.pt` (bool [T,H,W]). "
                        "When set, every video uses its own per-frame mask from this dir.")
    p.add_argument("--num_select", type=int, default=4)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--flow_cache_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flow_batch", type=int, default=8, help="Frame pairs per RAFT forward (RAFT only)")
    p.add_argument("--video_glob", type=str, default="seed*.mp4")
    p.add_argument("--flow_backend", type=str, default="raft", choices=["raft", "memfof"],
                   help="Optical flow backend")
    p.add_argument("--memfof_model", type=str, default="MEMFOF-Tartan-T-TSKH",
                   help="MEMFOF checkpoint name (loaded via huggingface_hub)")
    p.add_argument("--memfof_iters", type=int, default=8, help="Refinement iterations for MEMFOF")
    p.add_argument("--no_mask", action="store_true",
                   help="Use all pixels (foreground + background) instead of a mask. "
                        "Overrides --mask_path and --mask_dir.")
    p.add_argument("--horizon", type=int, default=1,
                   help="Sliding window size for cumulative motion displacement. "
                        "1 (default) = instantaneous frame-to-frame flow. "
                        ">1 = cumulative displacement over the next N frames. "
                        "N=T-1 = full-video trajectory (drift-prone, not recommended).")
    return p.parse_args()


def find_videos(video_dir: Path, glob_pat: str):
    """Return list of (seed_int, path) sorted by seed."""
    paths = sorted(video_dir.glob(glob_pat))
    out = []
    for p in paths:
        m = SEED_RE.search(p.name)
        if m is None:
            print(f"WARN: skipping {p.name} — no seed in name", file=sys.stderr)
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_video_frames(path: Path) -> torch.Tensor:
    """Decode an mp4 with PyVideoReader. Returns uint8 tensor [T, 3, H, W]."""
    vr = PyVideoReader(str(path), threads=0)
    buf = vr.decode()  # numpy [T, H, W, 3] uint8
    del vr
    frames = torch.from_numpy(buf).permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]
    return frames


def load_and_resize_mask(mask_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load SAM2-style mask (255=fg, 0=bg) and resize to (target_h, target_w).
    Returns bool tensor [target_h, target_w] where True = foreground."""
    img = Image.open(mask_path).convert("L")
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.NEAREST)
    arr = np.array(img)
    return torch.from_numpy(arr > 127)


@torch.no_grad()
def compute_flow_raft(model, frames: torch.Tensor, device: str, batch: int) -> torch.Tensor:
    """frames: uint8 [T, 3, H, W]. Returns float32 [2, T-1, H, W] flow on CPU.

    RAFT expects float in [-1, 1]. We feed consecutive (frame_i, frame_{i+1}) pairs.
    """
    T = frames.shape[0]
    if T < 2:
        raise ValueError(f"Video too short: {T} frames")
    f = frames.float() / 127.5 - 1.0  # [T, 3, H, W] in [-1, 1]
    flows = []
    for s in range(0, T - 1, batch):
        e = min(s + batch, T - 1)
        a = f[s:e].to(device, non_blocking=True)
        b = f[s + 1 : e + 1].to(device, non_blocking=True)
        # raft_large returns a list of refined flows; the last is the final estimate
        out = model(a, b)
        flow = out[-1] if isinstance(out, (list, tuple)) else out  # [B, 2, H, W]
        flows.append(flow.cpu())
    flow = torch.cat(flows, dim=0)  # [T-1, 2, H, W]
    return flow.permute(1, 0, 2, 3).contiguous()  # [2, T-1, H, W]


@torch.no_grad()
def compute_flow_memfof(model, frames: torch.Tensor, device: str, iters: int) -> torch.Tensor:
    """frames: uint8 [T, 3, H, W] in [0,255]. Returns float32 [2, T-1, H, W] flow on CPU.

    MEMFOF takes triplets [B, 3, 3, H, W] and returns both backward and forward
    flows around the *middle* frame. We slide a 3-frame window and capture:
      - From the FIRST triplet (0,1,2): both the (negated) backward flow [:, 0]
        for pair (0->1) AND the forward flow [:, 1] for pair (1->2).
      - From every subsequent triplet (i,i+1,i+2): only the forward flow [:, 1]
        for pair (i+1 -> i+2).
    Total: T-1 forward-equivalent flows (matches RAFT shape).

    fmap_cache is reused across triplets for efficiency (per the demo).
    """
    T = frames.shape[0]
    if T < 3:
        raise ValueError(f"MEMFOF needs >=3 frames, got {T}")

    f = frames.float()  # MEMFOF normalises internally (expects [0,255])
    out_flows = []  # list of [2, H, W] tensors on CPU
    fmap_cache = [None, None, None]
    first = True

    for i in range(T - 2):
        triplet = f[i:i + 3].unsqueeze(0).to(device)  # [1, 3, 3, H, W]
        result = model(triplet, iters=iters, fmap_cache=fmap_cache)
        flow = result["flow"][-1]  # [1, 2 (BW,FW), 2, H, W]

        if first:
            # backward flow [:, 0] is the flow from middle (frame 1) -> past (frame 0).
            # We want pair flow 0->1, which is the inverse direction. Negate to get it.
            bw = flow[:, 0]  # [1, 2, H, W]
            out_flows.append((-bw[0]).cpu())
            first = False

        fw = flow[:, 1]  # forward flow: middle (frame i+1) -> future (frame i+2)
        out_flows.append(fw[0].cpu())

        # slide cache one step (per demo)
        fmap_cache = result["fmap_cache"]
        fmap_cache.pop(0)
        fmap_cache.append(None)

    flow = torch.stack(out_flows, dim=1)  # [2, T-1, H, W]
    return flow.contiguous()


@torch.no_grad()
def compute_horizon_displacement(flow: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding-window cumulative displacement with per-pixel visibility tracking.

    flow: [2, T-1, H, W] frame-to-frame pixel-unit (u, v) displacements.
    horizon: int >= 1. Window size in frames.

    Returns:
      disp [2, T-horizon, H, W]: pixel-unit displacement of the pixel
        originally at (y, x) at frame `t_start`, accumulated over the next
        `horizon` frames.
      vis  [T-horizon, H, W] bool: True if that pixel's integrated position
        remained in [0, W) × [0, H) at every step within the window.

    Drift is bounded by O(σ·√horizon), where σ is the per-frame RAFT error,
    so a small horizon (e.g., 4-8) recovers MotionModes-flavored multi-frame
    motion patterns while keeping integration noise negligible. horizon=1 is
    exactly the input flow (instantaneous velocity); pixel that go out of
    frame in one step are flagged via vis.

    Why visibility instead of clamp: MotionModes' loss uses clamp(-1, 1) on
    a [0, 1]-normalized flow, which homogenizes off-screen pixels into the
    boundary value. For raw pixel displacements we exclude such pixels
    entirely — fewer ghost contributions to the pairwise distance.
    """
    _, T1, H, W = flow.shape
    device = flow.device
    dtype = flow.dtype
    if horizon < 1 or horizon > T1:
        raise ValueError(f"horizon must be in [1, {T1}], got {horizon}")

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    if horizon == 1:
        # 1-frame displacement is the flow itself; visibility = "destination is in-frame".
        end_x = xx.unsqueeze(0) + flow[0]  # [T-1, H, W]
        end_y = yy.unsqueeze(0) + flow[1]
        vis = (end_x >= 0) & (end_x <= W - 1) & (end_y >= 0) & (end_y <= H - 1)
        return flow, vis

    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)

    # Forward-integrate the full per-frame trajectory of every pixel.
    pos_x = xx.clone()
    pos_y = yy.clone()
    pos_x_hist = [pos_x.clone()]
    pos_y_hist = [pos_y.clone()]
    in_frame_hist = [torch.ones(H, W, device=device, dtype=torch.bool)]

    for t in range(T1):
        gx = sx * pos_x - 1.0
        gy = sy * pos_y - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            flow[:, t].unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        pos_x = pos_x + sampled[0, 0]
        pos_y = pos_y + sampled[0, 1]
        pos_x_hist.append(pos_x.clone())
        pos_y_hist.append(pos_y.clone())
        in_frame_hist.append((pos_x >= 0) & (pos_x <= W - 1) & (pos_y >= 0) & (pos_y <= H - 1))

    pos_x_stack = torch.stack(pos_x_hist, dim=0)  # [T1+1, H, W]
    pos_y_stack = torch.stack(pos_y_hist, dim=0)
    in_frame_stack = torch.stack(in_frame_hist, dim=0)

    out_frames = T1 - horizon + 1  # = T - horizon (since T1 = T - 1)
    disp_x = pos_x_stack[horizon:] - pos_x_stack[:out_frames]  # [out_frames, H, W]
    disp_y = pos_y_stack[horizon:] - pos_y_stack[:out_frames]
    disp = torch.stack([disp_x, disp_y], dim=0).contiguous()  # [2, out_frames, H, W]

    # Visibility: AND over all frames within the window [t_start, t_start + horizon].
    vis = torch.empty(out_frames, H, W, device=device, dtype=torch.bool)
    for t in range(out_frames):
        vis[t] = in_frame_stack[t : t + horizon + 1].all(dim=0)

    return disp, vis


def pairwise_distance(f_i: torch.Tensor, f_j: torch.Tensor,
                      mask_fg: torch.Tensor | None = None,
                      eps: float = 1e-8) -> float:
    """Symmetric distance between two flow tensors [2, T, H, W].

    When `mask_fg` is None, all pixels are used. Otherwise:
      - 2D `[H, W]`: same static mask is broadcast across all T frames
      - 3D `[T, H, W]`: per-frame mask (time-varying tracking)

    Mirrors MotionModes/flowgen/motion_control/motion_losses_ours.py:991-1043
    (pairwise_difference_loss_angle_mag) but:
      - takes two flows directly (not flow-vs-list)
      - returns the raw avg_combined_diff (larger = more different),
        i.e. *not* the inverse-loss form
      - skips the `flow*2 - 1` rescale from L1006 because RAFT outputs are
        pixel displacements, not [0,1]-normalized
    """
    # Magnitude difference, normalized per-component by joint magnitude
    diff = (f_i - f_j).abs()  # [2, T, H, W]
    mag = torch.sqrt(f_i.pow(2) + f_j.pow(2) + eps)
    normalized_diff = diff / (mag + eps)  # [2, T, H, W]

    # Angular difference via cosine similarity of the (u, v) vectors at each pixel
    f_i_n = F.normalize(f_i, dim=0)  # normalize over the 2-vector axis
    f_j_n = F.normalize(f_j, dim=0)
    cos_sim = (f_i_n * f_j_n).sum(dim=0)  # [T, H, W]
    angle_diff = torch.acos(cos_sim.clamp(-1.0 + eps, 1.0 - eps))

    combined = normalized_diff.mean(dim=0) + angle_diff  # [T, H, W]

    if mask_fg is None:
        return combined.mean().item()

    # Build a [T, H, W] foreground selector matching `combined`
    if mask_fg.dim() == 2:
        fg = mask_fg.unsqueeze(0).expand_as(combined)
    elif mask_fg.dim() == 3:
        # Time-varying mask. Slice / broadcast to T frames if shapes mismatch.
        if mask_fg.shape[0] == combined.shape[0]:
            fg = mask_fg
        elif mask_fg.shape[0] == combined.shape[0] + 1:
            # Mask was made on T frames, flow has T-1 frame pairs.
            # Use the union of consecutive frames so a pixel that is fg in either
            # endpoint is counted (avoids dropping the moving boundary).
            fg = mask_fg[:-1] | mask_fg[1:]
        else:
            raise ValueError(f"mask T={mask_fg.shape[0]} vs flow T={combined.shape[0]} mismatch")
    else:
        raise ValueError(f"mask_fg must be 2D or 3D, got {mask_fg.shape}")
    n_valid = int(fg.sum())
    if n_valid == 0:
        return float("nan")
    return combined[fg].mean().item()


def farthest_point_sampling(D: np.ndarray, k: int) -> list:
    """Greedy FPS on a precomputed symmetric distance matrix. Returns list of indices.

    Anchor: index with largest sum of distances to all others (most novel point).
    Then iteratively pick argmax of min-distance to selected.
    """
    n = D.shape[0]
    if k >= n:
        return list(range(n))
    selected = [int(np.argmax(D.sum(axis=1)))]
    min_d = D[selected[0]].copy()
    for _ in range(1, k):
        # exclude already-selected
        cand = min_d.copy()
        cand[selected] = -np.inf
        nxt = int(np.argmax(cand))
        selected.append(nxt)
        min_d = np.minimum(min_d, D[nxt])
    return selected


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.flow_cache_dir) if args.flow_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_dir, args.video_glob)
    if not videos:
        sys.exit(f"No videos matched {args.video_glob} in {video_dir}")
    print(f"Found {len(videos)} videos (seed range {videos[0][0]}..{videos[-1][0]})")

    # Probe first video for resolution
    probe = load_video_frames(videos[0][1])
    T0, _, H, W = probe.shape
    print(f"Video shape: T={T0}, H={H}, W={W}")
    del probe

    # Resolve mask source: --no_mask, --mask_dir (time-varying), or --mask_path (static)
    use_no_mask = args.no_mask
    use_time_varying_mask = (not use_no_mask) and (args.mask_dir is not None)
    static_mask_fg = None
    mask_dir = Path(args.mask_dir) if use_time_varying_mask else None
    if use_no_mask:
        print("Using all pixels (no mask) for distance computation")
    elif use_time_varying_mask:
        # sanity check: make sure every video has a corresponding mask cache
        missing = [s for s, _ in videos if not (mask_dir / f"seed{s}.pt").exists()]
        if missing:
            sys.exit(f"Missing per-video masks in {mask_dir}: seeds={missing[:5]}... ({len(missing)} total)")
        print(f"Using per-video time-varying masks from {mask_dir}")
        # Save the first-frame mask of the first video as a verification artifact
        sample = torch.load(mask_dir / f"seed{videos[0][0]}.pt", map_location="cpu", weights_only=True)
        Image.fromarray((sample[0].numpy().astype(np.uint8) * 255)).save(out_dir / "mask_used.png")
        print(f"Sample mask shape (seed {videos[0][0]}): {tuple(sample.shape)}, mean fg={sample.float().mean():.3f}")
    else:
        if args.mask_path is None:
            sys.exit("Either --no_mask, --mask_dir, or --mask_path must be provided.")
        static_mask_fg = load_and_resize_mask(Path(args.mask_path), H, W)
        fg_count = int(static_mask_fg.sum())
        print(f"Static mask foreground pixels: {fg_count}/{H*W} ({fg_count/(H*W):.2%})")
        if fg_count == 0:
            sys.exit("Mask has no foreground pixels after resize.")
        Image.fromarray((static_mask_fg.numpy().astype(np.uint8) * 255)).save(out_dir / "mask_used.png")

    # Load optical flow model
    if args.flow_backend == "raft":
        print("Loading torchvision RAFT-large...")
        weights = Raft_Large_Weights.C_T_SKHT_V2
        model = raft_large(weights=weights, progress=True).to(args.device).eval()
    elif args.flow_backend == "memfof":
        print(f"Loading MEMFOF ({args.memfof_model})...")
        from memfof.model import MEMFOF
        model = MEMFOF.from_pretrained(f"egorchistov/optical-flow-{args.memfof_model}").to(args.device).eval()
    else:
        sys.exit(f"Unknown flow backend: {args.flow_backend}")

    # Compute flows (cache-aware) — cache filenames include backend tag
    flows = []  # list of [2, T-1, H, W] CPU float32
    for seed, path in tqdm(videos, desc="flow"):
        cache_file = (cache_dir / f"seed{seed}.pt") if cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            flow = torch.load(cache_file, map_location="cpu", weights_only=True)
        else:
            frames = load_video_frames(path)
            if args.flow_backend == "raft":
                flow = compute_flow_raft(model, frames, args.device, args.flow_batch)
            else:
                flow = compute_flow_memfof(model, frames, args.device, args.memfof_iters)
            if cache_file is not None:
                torch.save(flow, cache_file)
            del frames
        flows.append(flow)

    # Free GPU model — distances are computed on flows directly
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Build motion representation = horizon-window cumulative displacement + visibility.
    # horizon=1 returns the flow itself with a destination in-frame check.
    # horizon>1 integrates and emits sliding-window displacements; pixels that exit
    # the frame within the window are marked invalid (no clamp homogenization).
    horizon = args.horizon
    T_motion = (flows[0].shape[1] + 1) - horizon  # T - horizon = (T-1) + 1 - horizon
    print(f"Horizon: {horizon} frames  (per-video motion shape: [2, {T_motion}, {H}, {W}])")
    device = torch.device(args.device)
    motion = []     # list of [2, T-horizon, H, W] CPU float32
    visibility = [] # list of [T-horizon, H, W] CPU bool
    for f in tqdm(flows, desc="motion"):
        f_dev = f.to(device)
        disp, vis = compute_horizon_displacement(f_dev, horizon)
        motion.append(disp.cpu())
        visibility.append(vis.cpu())
        del f_dev, disp, vis
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    del flows

    # Reduce user foreground mask to per-video temporal mask of length T-horizon.
    # - no_mask: per-video foreground = all True (visibility alone gates the pixels)
    # - static mask: broadcast 2D to all T-horizon frames
    # - time-varying mask [T, H, W]: per-window OR-reduce over (horizon+1) frames
    def _reduce_time_varying(m: torch.Tensor, h: int) -> torch.Tensor:
        # m: [T, H, W] bool. Returns [T - h, H, W] = OR over m[t : t + h + 1].
        T_orig = m.shape[0]
        if T_orig - h != T_motion:
            raise ValueError(
                f"mask T={T_orig} incompatible with horizon={h}; need T - h == {T_motion}"
            )
        out = torch.empty(T_motion, *m.shape[1:], device=m.device, dtype=torch.bool)
        for t in range(T_motion):
            out[t] = m[t : t + h + 1].any(dim=0)
        return out

    # Combine visibility AND foreground mask into a single [T-horizon, H, W] bool
    # per video, then move to GPU for distance computation.
    print("Building per-video masks (visibility ∧ foreground)...")
    combined_masks_dev = []
    for idx, (seed, _) in enumerate(videos):
        vis = visibility[idx]  # [T-horizon, H, W] bool
        if use_no_mask:
            fg = vis  # visibility alone gates
        elif use_time_varying_mask:
            m = torch.load(mask_dir / f"seed{seed}.pt", map_location="cpu", weights_only=True)
            fg_user = _reduce_time_varying(m, horizon)
            fg = vis & fg_user
        else:
            fg = vis & static_mask_fg.unsqueeze(0)  # broadcast 2D
        combined_masks_dev.append(fg.to(device))
    del visibility

    # Pairwise distance matrix on GPU
    N = len(motion)
    print(f"Computing {N}x{N} distance matrix...")
    motion_dev = [m.to(device) for m in motion]

    D = np.zeros((N, N), dtype=np.float64)
    for i in tqdm(range(N), desc="dist"):
        for j in range(i + 1, N):
            pair_mask = combined_masks_dev[i] & combined_masks_dev[j]
            d = pairwise_distance(motion_dev[i], motion_dev[j], pair_mask)
            D[i, j] = d
            D[j, i] = d

    np.save(out_dir / "distance_matrix.npy", D)
    print(f"Distance matrix: min={D[np.triu_indices(N, k=1)].min():.4f} "
          f"mean={D[np.triu_indices(N, k=1)].mean():.4f} "
          f"max={D[np.triu_indices(N, k=1)].max():.4f}")

    # Per-video valid pixel counts (useful diagnostic, esp. for larger horizons)
    valid_frac = np.array([float(m.float().mean().item()) for m in combined_masks_dev])

    # Free motion + mask tensors on GPU
    del motion_dev, combined_masks_dev
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Farthest Point Sampling
    selected_idx = farthest_point_sampling(D, args.num_select)
    selected_seeds = [videos[i][0] for i in selected_idx]
    print(f"Selected indices: {selected_idx}")
    print(f"Selected seeds:   {selected_seeds}")

    # Write outputs: symlink (fall back to copy)
    rank_to_seed = {}
    for rank, idx in enumerate(selected_idx):
        seed, src = videos[idx]
        dst = out_dir / f"rank{rank}_seed{seed}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy(src, dst)
        rank_to_seed[rank] = seed

    # selected.json with summary stats
    sub = D[np.ix_(selected_idx, selected_idx)]
    iu = np.triu_indices(len(selected_idx), k=1)
    summary = {
        "horizon": args.horizon,
        "flow_backend": args.flow_backend,
        "mask_mode": "no_mask" if use_no_mask else ("time_varying" if use_time_varying_mask else "static"),
        "valid_pixel_frac_mean": float(valid_frac.mean()),
        "valid_pixel_frac_min": float(valid_frac.min()),
        "num_videos_total": N,
        "num_selected": len(selected_idx),
        "selected_indices": selected_idx,
        "selected_seeds": selected_seeds,
        "rank_to_seed": rank_to_seed,
        "selected_min_pairwise_distance": float(sub[iu].min()) if len(iu[0]) else None,
        "selected_mean_pairwise_distance": float(sub[iu].mean()) if len(iu[0]) else None,
        "global_min_pairwise_distance": float(D[np.triu_indices(N, k=1)].min()),
        "global_mean_pairwise_distance": float(D[np.triu_indices(N, k=1)].mean()),
        "global_max_pairwise_distance": float(D[np.triu_indices(N, k=1)].max()),
    }
    (out_dir / "selected.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir/'selected.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
