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

Selection (`--selector`):
  - `fps` (default): Farthest Point Sampling on the pairwise distance matrix.
    Pure binary-term selection; no per-video quality score.
  - `qip`: group-inference-style quadratic objective
        max  Σᵢ uᵢ·xᵢ  +  λ · Σᵢ<ⱼ D[i,j]·xᵢ·xⱼ        s.t. Σ xᵢ = K
    Solved by marginal-gain greedy (no Gurobi dependency). u and D are each
    min-max normalized to [0, 1] before combining, so λ controls the
    diversity/quality trade-off on a predictable scale. Mirrors
    group-inference/src/my_utils/solvers.py:gurobi_solver up to the QIP→greedy
    relaxation.

Unary (`--unary`):
  - `none` (default): no per-video quality term.
  - `clip`: CLIP text-image cosine similarity averaged over uniformly-sampled
    frames per video (requires `--prompt`). Lifts the unary from
    group-inference/src/my_utils/scores.py:unary_clip_text_img_t to video.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
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
    p.add_argument("--raft_updates", type=int, default=8,
                   help="RAFT num_flow_updates (refinement iterations). Default 8 (~1.5x faster "
                        "than the torchvision default 12, negligible quality loss). 6 is also "
                        "acceptable; below that quality drops visibly.")
    p.add_argument("--raft_dtype", type=str, default="bf16", choices=["bf16", "fp32"],
                   help="RAFT compute dtype. bf16 (default) uses torch.autocast and is ~1.5x "
                        "faster than fp32 with no measurable quality drop on Helios-sized inputs.")
    p.add_argument("--video_glob", type=str, default="seed*.mp4")
    p.add_argument("--flow_backend", type=str, default="raft", choices=["raft", "memfof"],
                   help="Optical flow backend")
    p.add_argument("--memfof_model", type=str, default="MEMFOF-Tartan-T-TSKH",
                   help="MEMFOF checkpoint name (loaded via huggingface_hub)")
    p.add_argument("--memfof_iters", type=int, default=8, help="Refinement iterations for MEMFOF")
    p.add_argument("--no_mask", action="store_true",
                   help="Use all pixels (foreground + background) instead of a mask. "
                        "Overrides --mask_path and --mask_dir.")
    p.add_argument("--motion_threshold", type=float, default=0.0,
                   help="Per-pixel motion-magnitude gate (in pixels). If >0, only pixels whose "
                        "displacement magnitude exceeds this threshold in at least one of the "
                        "paired videos are scored. Effectively a SAM2-free way to focus on "
                        "moving regions. Applied per (t, y, x). Stacks with --mask_dir if both "
                        "are set.")
    p.add_argument("--horizon", type=int, default=1,
                   help="Sliding window size for cumulative motion displacement. "
                        "1 (default) = instantaneous frame-to-frame flow. "
                        ">1 = cumulative displacement over the next N frames. "
                        "N=T-1 = full-video trajectory (drift-prone, not recommended).")
    p.add_argument("--selector", type=str, default="fps", choices=["fps", "qip"],
                   help="Selection algorithm. 'fps' (default) = pairwise-only Farthest Point "
                        "Sampling. 'qip' = greedy approximation to the group-inference "
                        "quadratic objective max Σuᵢxᵢ + λΣᵢ<ⱼD[i,j]xᵢxⱼ.")
    p.add_argument("--unary", type=str, default="none", choices=["none", "clip"],
                   help="Per-video unary quality score. 'clip' = CLIP text-image cosine "
                        "similarity averaged over uniformly-sampled frames (requires --prompt). "
                        "When --selector=fps, the unary is only logged, not used for selection.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt for CLIP unary scoring. Required when --unary=clip.")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                   help="HF model id for CLIP unary (matches group-inference default).")
    p.add_argument("--clip_frames", type=int, default=8,
                   help="Number of frames to uniformly sample per video for CLIP unary.")
    p.add_argument("--unary_cache_dir", type=str, default=None,
                   help="Cache directory for unary scores. Keyed by (model, num_frames, prompt).")
    p.add_argument("--lambda_score", type=float, default=1.0,
                   help="Weight on the binary (pairwise diversity) term in the QIP objective. "
                        "Applied after min-max normalizing both u and D to [0, 1].")
    p.add_argument("--trajectory_cache_dir", type=str, default=None,
                   help="Directory of pre-computed dense trajectories `seed{N}.pt` [2, T-1, H, W] "
                        "in absolute pixel coords (e.g., output of tools/compute_dot_trajectories.py). "
                        "When set, skips flow computation and integration — uses the trajectories "
                        "directly. Drift-free since DOT outputs accurate long-range tracks.")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Truncate each video to the first N frames before computing flow. "
                        "Use with separate --flow_cache_dir per frame count to avoid "
                        "cache collisions. Enables cascading selection across chunk boundaries.")
    p.add_argument("--seed_filter", type=str, default=None,
                   help="Comma-separated seed IDs to process (e.g., '3,16,22,42'). "
                        "Videos with seeds not in this list are skipped. Use to chain "
                        "cascading rounds: pass the previous round's selected_seeds here.")
    p.add_argument("--mm_mask_mode", type=str, default="frame0", choices=["frame0", "per_frame"],
                   help="(deprecated, unused) --mm_compat now always uses the SAM2 "
                        "time-varying mask, OR-reduced per horizon window.")
    p.add_argument("--mm_compat", action="store_true",
                   help="Use the MotionModes distance kernel (motion_losses_ours.py:991-1043): "
                        "normalize positions to MM 'flow' space (pos/dim, == (pos/dim+1)/2 then "
                        "*2-1) and clamp(-1,1), off-screen handled by clamp (not visibility "
                        "exclusion). Unlike the default path it does NOT use the principled "
                        "visibility mask. Honors --horizon (window difference in MM-normalized "
                        "space; --horizon 1 = instantaneous). Foreground = SAM2 time-varying mask.")
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


def load_video_frames(path: Path, max_frames: int | None = None) -> torch.Tensor:
    """Decode an mp4 with PyVideoReader. Returns uint8 tensor [T, 3, H, W]."""
    vr = PyVideoReader(str(path), threads=0)
    buf = vr.decode()  # numpy [T, H, W, 3] uint8
    del vr
    frames = torch.from_numpy(buf).permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]
    if max_frames is not None and frames.shape[0] > max_frames:
        frames = frames[:max_frames]
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
def compute_flow_raft(model, frames: torch.Tensor, device: str, batch: int,
                      num_flow_updates: int = 8,
                      compute_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """frames: uint8 [T, 3, H, W]. Returns float32 [2, T-1, H, W] flow on CPU.

    RAFT expects float in [-1, 1]. We feed consecutive (frame_i, frame_{i+1}) pairs.

    - `num_flow_updates` is RAFT's refinement iteration count; the torchvision default
      is 12. 8 cuts compute by ~1.5x with no measurable quality loss at Helios sizes.
    - `compute_dtype=bf16` uses torch.autocast for another ~1.5x. Cache output is
      always float32 for backward compatibility with existing cached flows.
    """
    T = frames.shape[0]
    if T < 2:
        raise ValueError(f"Video too short: {T} frames")
    # Upload the whole video to GPU once. For 97 frames at 384x640 this is ~280 MB
    # in fp32 — well within budget, and avoids re-uploading each pair's frame across
    # adjacent batches (every middle frame would otherwise be sent twice).
    f = (frames.float().to(device, non_blocking=True) / 127.5) - 1.0  # [T, 3, H, W] in [-1, 1]
    use_autocast = compute_dtype is not torch.float32
    flows = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if use_autocast and str(device).startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast_ctx:
        for s in range(0, T - 1, batch):
            e = min(s + batch, T - 1)
            out = model(f[s:e], f[s + 1 : e + 1], num_flow_updates=num_flow_updates)
            flow = out[-1] if isinstance(out, (list, tuple)) else out  # [B, 2, H, W]
            # Cast back to fp32 so the cache stays compatible with prior runs.
            flows.append(flow.float().cpu())
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
def horizon_displacement_from_trajectory(traj: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Same output as compute_horizon_displacement, but the trajectory is supplied
    directly (e.g., from DOT) instead of integrated from frame-to-frame flow.

    traj: [2, T-1, H, W] absolute pixel positions of every origin pixel at frames
          1..T-1 (frame 0 is implicit identity).
    horizon: window size, must satisfy 1 <= horizon <= T-1.

    Returns (disp [2, T-horizon, H, W], vis [T-horizon, H, W] bool).
    Drift-free when traj comes from an accurate tracker like DOT.
    """
    _, T1, H, W = traj.shape
    device = traj.device
    dtype = traj.dtype
    if horizon < 1 or horizon > T1:
        raise ValueError(f"horizon must be in [1, {T1}], got {horizon}")

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    identity = torch.stack([xx, yy], dim=0)  # [2, H, W]

    # Prepend identity (frame 0 position) to make full [2, T, H, W] trajectory
    full = torch.cat([identity.unsqueeze(1), traj], dim=1)  # [2, T, H, W]

    out_frames = T1 - horizon + 1  # = T - horizon
    disp = (full[:, horizon:] - full[:, :out_frames]).contiguous()  # [2, out_frames, H, W]

    # Visibility: in-frame at every step within the window
    in_frame = (full[0] >= 0) & (full[0] <= W - 1) & (full[1] >= 0) & (full[1] <= H - 1)  # [T, H, W]
    vis = torch.empty(out_frames, H, W, device=device, dtype=torch.bool)
    for t in range(out_frames):
        vis[t] = in_frame[t : t + horizon + 1].all(dim=0)

    return disp, vis


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


def qip_greedy(u: np.ndarray, D: np.ndarray, k: int, lam: float, n_anchors: int | None = None) -> list:
    """Multi-restart greedy approximation to the group-inference QIP objective.

        maximize  Σᵢ uᵢ·xᵢ  +  λ · Σᵢ<ⱼ D[i,j]·xᵢ·xⱼ      s.t.  Σ xᵢ = k

    Marginal-gain greedy from a fixed first pick (anchor): at each later step pick
    argmax of `uᵢ + λ · Σ_{s∈S} D[i, s]`. Standard (1 − 1/e)-approximation for
    monotone submodular maximization with cardinality k (objective is monotone
    submodular when D ≥ 0 and u ≥ 0, which holds after [0, 1] min-max
    normalization in main()).

    The single-anchor variant locks in `argmax(u)` as the first pick and cannot
    recover from a suboptimal start — when the top-unary item sits next to other
    high-unary similar items, every later pick stays in that neighborhood
    regardless of λ. We mitigate this by running greedy from every candidate
    anchor and returning the configuration with the highest objective value.
    Cost is O(N · K · N) = O(KN²), trivial for the K ≤ 32, N ≤ few-thousand
    regime this script targets.

    n_anchors: number of distinct first-pick anchors to try. Default = N (try
    every item as anchor). Pass 1 to recover the original single-anchor greedy.

    Cf. group-inference/src/my_utils/solvers.py:gurobi_solver — same objective,
    exact via QIP. Greedy is used here to avoid a Gurobi dependency.
    """
    n = len(u)
    if k >= n:
        return list(range(n))
    if u.shape != (n,) or D.shape != (n, n):
        raise ValueError(f"shape mismatch: u={u.shape}, D={D.shape}")
    if n_anchors is None or n_anchors <= 0 or n_anchors > n:
        n_anchors = n

    u_f = u.astype(np.float64)
    D_f = D.astype(np.float64)
    candidate_anchors = np.argsort(-u_f)[:n_anchors].tolist()

    def _greedy_from(anchor: int) -> list:
        selected = [anchor]
        in_S = np.zeros(n, dtype=bool)
        in_S[anchor] = True
        bin_gain = D_f[anchor].copy()
        for _ in range(k - 1):
            gain = u_f + lam * bin_gain
            gain[in_S] = -np.inf
            i = int(np.argmax(gain))
            selected.append(i)
            in_S[i] = True
            bin_gain = bin_gain + D_f[i]
        return selected

    def _objective(sel: list) -> float:
        idx = np.array(sel)
        sub = D_f[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        return float(u_f[idx].sum() + lam * sub[iu].sum())

    best_sel: list = []
    best_obj = -np.inf
    for anchor in candidate_anchors:
        sel = _greedy_from(anchor)
        obj = _objective(sel)
        if obj > best_obj:
            best_obj = obj
            best_sel = sel
    return best_sel


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if range is degenerate."""
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return ((x - lo) / (hi - lo)).astype(np.float64)


@torch.no_grad()
def compute_unary_clip(
    videos: list,
    prompt: str,
    model_name: str,
    num_frames: int,
    cache_dir: Path | None,
    device: str,
) -> np.ndarray:
    """Per-video CLIP text-image cosine similarity, averaged over uniformly-sampled frames.

    Mirrors group-inference/src/my_utils/scores.py:unary_clip_text_img_t but
    lifted to video: decode each video, uniformly sample `num_frames`, compute
    CLIP image features, take cos-sim with the text embedding, average. Returns
    a float32 array of length N (cosine similarities, roughly in [0, 1] after
    normalization but not strictly clamped).

    Cache key is hashed from (model_name, num_frames, prompt) so changing any
    of those forces recomputation.
    """
    from transformers import CLIPModel, CLIPProcessor

    n = len(videos)
    out = np.zeros(n, dtype=np.float32)

    key = hashlib.md5(f"{model_name}|{num_frames}|{prompt}".encode("utf-8")).hexdigest()[:10]
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"key_{key}.txt").write_text(
            json.dumps({"model": model_name, "num_frames": num_frames, "prompt": prompt}, indent=2)
        )

    print(f"Loading CLIP ({model_name})...")
    m_clip = CLIPModel.from_pretrained(model_name).to(device).eval()
    prep = CLIPProcessor.from_pretrained(model_name)

    def _as_tensor(out):
        # Some transformers versions return a model-output object instead of a tensor.
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(out, attr, None)
            if v is not None:
                return v[:, 0] if attr == "last_hidden_state" else v
        raise TypeError(f"Unexpected CLIP output type: {type(out)}")

    text_in = prep.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    text_emb = F.normalize(_as_tensor(m_clip.get_text_features(**text_in)), dim=-1)  # [1, D]

    img_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    _t_clip0 = time.time()
    for idx, (seed, path) in enumerate(tqdm(videos, desc="clip")):
        cache_file = (cache_dir / f"seed{seed}_{key}.npy") if cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            out[idx] = float(np.load(cache_file).item())
            continue

        frames = load_video_frames(path)  # uint8 [T, 3, H, W]
        T = frames.shape[0]
        if T < num_frames:
            sel_idx = list(range(T))
        else:
            sel_idx = torch.linspace(0, T - 1, num_frames).round().long().tolist()
        sel = frames[sel_idx].to(device, non_blocking=True).float() / 255.0
        sel = F.interpolate(sel, size=(224, 224), mode="bilinear", align_corners=False)
        sel = (sel - img_mean) / img_std
        img_emb = F.normalize(_as_tensor(m_clip.get_image_features(pixel_values=sel)), dim=-1)  # [N, D]
        sim = (img_emb @ text_emb.T).squeeze(-1)  # [N]
        score = float(sim.mean().item())
        out[idx] = score
        if cache_file is not None:
            np.save(cache_file, np.array(score, dtype=np.float32))
        del frames, sel, img_emb, sim

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    print(f"[TIME] clip_unary_forward = {time.time() - _t_clip0:.2f}s  (excl CLIP model load)")
    del m_clip
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


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

    if args.seed_filter is not None:
        allowed = set(int(s) for s in args.seed_filter.split(","))
        videos = [(seed, p) for seed, p in videos if seed in allowed]
        if not videos:
            sys.exit(f"No videos left after --seed_filter (allowed={sorted(allowed)})")

    print(f"Found {len(videos)} videos (seed range {videos[0][0]}..{videos[-1][0]})")

    max_frames = args.max_frames
    # Probe first video for resolution (respecting max_frames)
    probe = load_video_frames(videos[0][1], max_frames=max_frames)
    T0, _, H, W = probe.shape
    print(f"Video shape: T={T0}, H={H}, W={W}" +
          (f" (truncated from full video via --max_frames={max_frames})" if max_frames else ""))
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

    # Either compute flows (then integrate to trajectory) or load pre-computed
    # drift-free trajectories (e.g., from DOT) and skip flow entirely.
    flows: list = []           # list of [2, T-1, H, W] CPU float32 (flow case)
    trajectories: list = []    # list of [2, T-1, H, W] CPU float32 absolute pixel coords (DOT case)
    if args.trajectory_cache_dir is not None:
        traj_dir = Path(args.trajectory_cache_dir)
        for seed, _ in tqdm(videos, desc="load_traj"):
            t = torch.load(traj_dir / f"seed{seed}.pt", map_location="cpu", weights_only=True)
            trajectories.append(t)
        print(f"Loaded {len(trajectories)} pre-computed trajectories from {traj_dir} (skipping flow)")
    else:
        _t_flow0 = time.time()
        # Enable TF32 (Ampere+) — a free ~10-20% speedup on RAFT/MEMFOF matmul.
        if str(args.device).startswith("cuda") and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load optical flow model
        if args.flow_backend == "raft":
            raft_dtype = torch.bfloat16 if args.raft_dtype == "bf16" else torch.float32
            print(f"Loading torchvision RAFT-large ({args.raft_dtype}, "
                  f"num_flow_updates={args.raft_updates})...")
            weights = Raft_Large_Weights.C_T_SKHT_V2
            model = raft_large(weights=weights, progress=True).to(args.device).eval()
        elif args.flow_backend == "memfof":
            print(f"Loading MEMFOF ({args.memfof_model})...")
            from memfof.model import MEMFOF
            model = MEMFOF.from_pretrained(f"egorchistov/optical-flow-{args.memfof_model}").to(args.device).eval()
        else:
            sys.exit(f"Unknown flow backend: {args.flow_backend}")

        # Compute flows (cache-aware) — cache filenames include backend tag
        _t_flowc0 = time.time()
        for seed, path in tqdm(videos, desc="flow"):
            cache_file = (cache_dir / f"seed{seed}.pt") if cache_dir is not None else None
            if cache_file is not None and cache_file.exists():
                flow = torch.load(cache_file, map_location="cpu", weights_only=True)
            else:
                frames = load_video_frames(path, max_frames=max_frames)
                if args.flow_backend == "raft":
                    flow = compute_flow_raft(model, frames, args.device, args.flow_batch,
                                              num_flow_updates=args.raft_updates,
                                              compute_dtype=raft_dtype)
                else:
                    flow = compute_flow_memfof(model, frames, args.device, args.memfof_iters)
                if cache_file is not None:
                    torch.save(flow, cache_file)
                del frames
            flows.append(flow)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[TIME] flow_compute = {time.time() - _t_flowc0:.2f}s  (excl RAFT model load)")

        # Free GPU model — distances are computed on flows directly
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[TIME] optical_flow_generation = {time.time() - _t_flow0:.2f}s")

    device = torch.device(args.device)
    _t_sel0 = time.time()

    # If pre-computed trajectories (or downsampled flows) have a different (H, W)
    # than the video, override H, W. Masks downstream are resized to match.
    using_traj = len(trajectories) > 0
    motion_source = trajectories[0] if using_traj else (flows[0] if len(flows) > 0 else None)
    if motion_source is not None:
        _, _, H_motion, W_motion = motion_source.shape
        if (H_motion, W_motion) != (H, W):
            kind = "trajectories" if using_traj else "flows"
            print(f"NOTE: {kind} at ({H_motion}, {W_motion}) differ from video at ({H}, {W}); "
                  f"masks will be resized to motion resolution.")
            H, W = H_motion, W_motion

    if args.mm_compat:
        # MotionModes kernel, honoring --horizon. Representation: integrate the
        # per-pixel trajectory, take the horizon-window displacement, normalize to
        # MotionModes "flow" space — pos/dim, i.e. MotionModes' (pos/dim+1)/2 followed
        # by the kernel's *2-1 (they cancel) — then clamp(-1, 1). Off-screen pixels are
        # handled by the clamp (not by visibility exclusion). Distance uses the same
        # magnitude+angle kernel; foreground = SAM2 time-varying mask, OR-reduced over
        # each (horizon+1)-frame window (no frame-0-only restriction).
        horizon = args.horizon
        T1_first = (trajectories[0] if using_traj else flows[0]).shape[1]
        T_motion = (T1_first + 1) - horizon
        print(f"Motion representation: MotionModes kernel (MM-normalized window diff), "
              f"horizon={horizon}  (per-video shape: [2, {T_motion}, {H}, {W}])")

        def _mm_normalize(disp: torch.Tensor) -> torch.Tensor:
            # raw-pixel window displacement -> MM flow space (pos/dim), clamped to [-1, 1]
            out = torch.empty_like(disp)
            out[0] = (disp[0] / W).clamp(-1.0, 1.0)
            out[1] = (disp[1] / H).clamp(-1.0, 1.0)
            return out

        motion = []  # list of [2, T-horizon, H, W] CPU float32 in [-1, 1]
        if using_traj:
            for t in tqdm(trajectories, desc="mm_horizon(traj)"):
                t_dev = t.to(device)
                disp, _ = horizon_displacement_from_trajectory(t_dev, horizon)
                motion.append(_mm_normalize(disp).cpu())
                del t_dev, disp
            del trajectories
        else:
            for f in tqdm(flows, desc="mm_horizon"):
                f_dev = f.to(device)
                disp, _ = compute_horizon_displacement(f_dev, horizon)
                motion.append(_mm_normalize(disp).cpu())
                del f_dev, disp
            del flows
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Foreground = SAM2 time-varying mask, OR-reduced over each window to T-horizon.
        def _resize_mask_mm(m: torch.Tensor) -> torch.Tensor:
            if m.shape[-2:] == (H, W):
                return m
            if m.dim() == 2:
                mf = m.float().unsqueeze(0).unsqueeze(0)
                mf = F.interpolate(mf, size=(H, W), mode="nearest")
                return mf.squeeze(0).squeeze(0).bool()
            mf = m.float().unsqueeze(1)
            mf = F.interpolate(mf, size=(H, W), mode="nearest")
            return mf.squeeze(1).bool()

        def _reduce_time_varying(m: torch.Tensor, h: int) -> torch.Tensor:
            # m: [T, H, W] bool -> [T - h, H, W], OR over each window m[t : t + h + 1].
            T_orig = m.shape[0]
            if T_orig - h != T_motion:
                raise ValueError(f"mask T={T_orig} incompatible with horizon={h}; need T-h=={T_motion}")
            out = torch.empty(T_motion, *m.shape[1:], device=m.device, dtype=torch.bool)
            for t in range(T_motion):
                out[t] = m[t : t + h + 1].any(dim=0)
            return out

        print("Building per-video masks (SAM2 time-varying, OR-reduced per window)...")
        combined_masks_dev = []
        for idx, (seed, _) in enumerate(videos):
            if use_no_mask:
                combined_masks_dev.append(None)
            elif use_time_varying_mask:
                m = torch.load(mask_dir / f"seed{seed}.pt", map_location="cpu", weights_only=True)
                if max_frames is not None and m.shape[0] > max_frames:
                    m = m[:max_frames]
                m = _resize_mask_mm(m)
                combined_masks_dev.append(_reduce_time_varying(m, horizon).to(device))
            else:
                static = _resize_mask_mm(static_mask_fg)
                combined_masks_dev.append(static.unsqueeze(0).expand(T_motion, H, W).contiguous().to(device))

        N = len(motion)
        print(f"Computing {N}x{N} distance matrix (MotionModes kernel, horizon={horizon})...")
        motion_dev = [m.to(device) for m in motion]

        motion_mag = None
        if args.motion_threshold > 0:
            motion_mag = [m.pow(2).sum(0).sqrt() for m in motion_dev]

        D = np.zeros((N, N), dtype=np.float64)
        for i in tqdm(range(N), desc="dist"):
            for j in range(i + 1, N):
                pair_mask = None
                if combined_masks_dev[i] is not None and combined_masks_dev[j] is not None:
                    pair_mask = combined_masks_dev[i] & combined_masks_dev[j]
                if motion_mag is not None:
                    gate = (motion_mag[i] > args.motion_threshold) | (motion_mag[j] > args.motion_threshold)
                    pair_mask = gate if pair_mask is None else (pair_mask & gate)
                d = pairwise_distance(motion_dev[i], motion_dev[j], pair_mask)
                D[i, j] = d
                D[j, i] = d
        if motion_mag is not None:
            del motion_mag

    else:
        # Build motion representation = horizon-window cumulative displacement + visibility.
        # horizon=1 returns the flow itself with a destination in-frame check.
        # horizon>1 integrates and emits sliding-window displacements; pixels that exit
        # the frame within the window are marked invalid (no clamp homogenization).
        # If trajectories were loaded, skip integration and just take sliding-window diffs.
        horizon = args.horizon
        if using_traj:
            T1_first = trajectories[0].shape[1]
        else:
            T1_first = flows[0].shape[1]
        T_motion = (T1_first + 1) - horizon  # T - horizon = (T-1) + 1 - horizon
        print(f"Horizon: {horizon} frames  (per-video motion shape: [2, {T_motion}, {H}, {W}])")
        motion = []     # list of [2, T-horizon, H, W] CPU float32
        visibility = [] # list of [T-horizon, H, W] CPU bool
        if using_traj:
            for t in tqdm(trajectories, desc="motion(traj)"):
                t_dev = t.to(device)
                disp, vis = horizon_displacement_from_trajectory(t_dev, horizon)
                motion.append(disp.cpu())
                visibility.append(vis.cpu())
                del t_dev, disp, vis
            del trajectories
        else:
            for f in tqdm(flows, desc="motion"):
                f_dev = f.to(device)
                disp, vis = compute_horizon_displacement(f_dev, horizon)
                motion.append(disp.cpu())
                visibility.append(vis.cpu())
                del f_dev, disp, vis
            del flows
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

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

        # Helper: resize a [T, H, W] bool mask to (H, W) (motion resolution) if needed.
        def _resize_mask(m: torch.Tensor) -> torch.Tensor:
            if m.shape[-2:] == (H, W):
                return m
            mf = m.float().unsqueeze(1)  # [T, 1, H_orig, W_orig]
            mf = F.interpolate(mf, size=(H, W), mode="nearest")
            return mf.squeeze(1).bool()

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
                if max_frames is not None and m.shape[0] > max_frames:
                    m = m[:max_frames]
                m = _resize_mask(m)
                fg_user = _reduce_time_varying(m, horizon)
                fg = vis & fg_user
            else:
                static_resized = _resize_mask(static_mask_fg.unsqueeze(0)).squeeze(0)
                fg = vis & static_resized.unsqueeze(0)  # broadcast 2D
            combined_masks_dev.append(fg.to(device))
        del visibility

        # Pairwise distance matrix on GPU
        N = len(motion)
        print(f"Computing {N}x{N} distance matrix...")
        motion_dev = [m.to(device) for m in motion]

        # Per-video per-pixel motion magnitudes for the optional motion gate.
        # motion_mag[i][t, y, x] = ||disp_i[:, t, y, x]||
        motion_mag = None
        if args.motion_threshold > 0:
            motion_mag = [m.pow(2).sum(0).sqrt() for m in motion_dev]
            all_mag = torch.cat([mg.flatten() for mg in motion_mag])
            print(f"Motion threshold: τ={args.motion_threshold:.2f} px. "
                  f"All-pixel motion magnitude: "
                  f"median={all_mag.median().item():.2f}, "
                  f"frac > τ = {(all_mag > args.motion_threshold).float().mean().item():.3f}")
            del all_mag

        D = np.zeros((N, N), dtype=np.float64)
        _t_dist0 = time.time()
        for i in tqdm(range(N), desc="dist"):
            for j in range(i + 1, N):
                pair_mask = combined_masks_dev[i] & combined_masks_dev[j]
                if motion_mag is not None:
                    # Gate: at least one of the two videos moves enough at this (t, y, x)
                    gate = (motion_mag[i] > args.motion_threshold) | (motion_mag[j] > args.motion_threshold)
                    pair_mask = pair_mask & gate
                d = pairwise_distance(motion_dev[i], motion_dev[j], pair_mask)
                D[i, j] = d
                D[j, i] = d
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[TIME] distance_matrix = {time.time() - _t_dist0:.2f}s")
        if motion_mag is not None:
            del motion_mag

    np.save(out_dir / "distance_matrix.npy", D)
    print(f"Distance matrix: min={D[np.triu_indices(N, k=1)].min():.4f} "
          f"mean={D[np.triu_indices(N, k=1)].mean():.4f} "
          f"max={D[np.triu_indices(N, k=1)].max():.4f}")

    # Per-video valid pixel counts (useful diagnostic, esp. for larger horizons)
    valid_frac = np.array(
        [1.0 if m is None else float(m.float().mean().item()) for m in combined_masks_dev]
    )

    # Free motion + mask tensors on GPU
    del motion_dev, combined_masks_dev
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Optional unary (per-video quality score). Currently CLIP text-image only.
    unary_scores = None
    if args.unary == "clip":
        if args.prompt is None:
            sys.exit("--unary=clip requires --prompt.")
        unary_cache = Path(args.unary_cache_dir) if args.unary_cache_dir else None
        unary_scores = compute_unary_clip(
            videos=videos,
            prompt=args.prompt,
            model_name=args.clip_model,
            num_frames=args.clip_frames,
            cache_dir=unary_cache,
            device=args.device,
        )
        print(f"CLIP unary scores: min={unary_scores.min():.4f} "
              f"mean={unary_scores.mean():.4f} max={unary_scores.max():.4f}")

    # Selection
    _t_alg0 = time.time()
    if args.selector == "fps":
        if unary_scores is not None:
            print("NOTE: --unary scores are computed and logged, but --selector=fps "
                  "ignores them. Use --selector=qip to combine.")
        selected_idx = farthest_point_sampling(D, args.num_select)
    elif args.selector == "qip":
        u_vec = unary_scores if unary_scores is not None else np.zeros(N, dtype=np.float64)
        u_norm = _minmax_normalize(u_vec)
        D_norm = _minmax_normalize(D)
        selected_idx = qip_greedy(u_norm, D_norm, args.num_select, args.lambda_score)
        print(f"QIP greedy: λ={args.lambda_score}, "
              f"unary={'clip' if unary_scores is not None else 'zero'}")
    else:
        sys.exit(f"Unknown selector: {args.selector}")
    print(f"[TIME] select_algo = {time.time() - _t_alg0:.3f}s")

    selected_seeds = [videos[i][0] for i in selected_idx]
    print(f"Selected indices: {selected_idx}")
    print(f"Selected seeds:   {selected_seeds}")
    print(f"[TIME] selection = {time.time() - _t_sel0:.2f}s")

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
        "mm_compat": bool(args.mm_compat),
        "horizon": args.horizon,
        "motion_threshold": args.motion_threshold,
        "flow_backend": args.flow_backend,
        "mask_mode": "no_mask" if use_no_mask else ("time_varying" if use_time_varying_mask else "static"),
        "valid_pixel_frac_mean": float(valid_frac.mean()),
        "valid_pixel_frac_min": float(valid_frac.min()),
        "num_videos_total": N,
        "num_selected": len(selected_idx),
        "selector": args.selector,
        "unary": args.unary,
        "lambda_score": args.lambda_score if args.selector == "qip" else None,
        "prompt": args.prompt if args.unary == "clip" else None,
        "clip_model": args.clip_model if args.unary == "clip" else None,
        "clip_frames": args.clip_frames if args.unary == "clip" else None,
        "selected_indices": selected_idx,
        "selected_seeds": selected_seeds,
        "rank_to_seed": rank_to_seed,
        "selected_min_pairwise_distance": float(sub[iu].min()) if len(iu[0]) else None,
        "selected_mean_pairwise_distance": float(sub[iu].mean()) if len(iu[0]) else None,
        "global_min_pairwise_distance": float(D[np.triu_indices(N, k=1)].min()),
        "global_mean_pairwise_distance": float(D[np.triu_indices(N, k=1)].mean()),
        "global_max_pairwise_distance": float(D[np.triu_indices(N, k=1)].max()),
    }
    if unary_scores is not None:
        summary["unary_scores_min"] = float(unary_scores.min())
        summary["unary_scores_mean"] = float(unary_scores.mean())
        summary["unary_scores_max"] = float(unary_scores.max())
        summary["selected_unary_scores"] = [float(unary_scores[i]) for i in selected_idx]
    (out_dir / "selected.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir/'selected.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
