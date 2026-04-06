"""
Select the K most motion-diverse videos from a directory by computing optical flow
and pairwise distances inside a mask region. Reference distance kernel:
MotionModes/flowgen/motion_control/motion_losses_ours.py:991-1043
(pairwise_difference_loss_angle_mag) — adapted into a symmetric pairwise distance.

Selection strategy: Farthest Point Sampling (FPS) — the K-extension of the
greedy 1-of-N diversification used in the MotionModes group inference loop.
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


SEED_RE = re.compile(r"seed(\d+)_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--mask_path", type=str, required=True)
    p.add_argument("--num_select", type=int, default=4)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--flow_cache_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flow_batch", type=int, default=8, help="Frame pairs per RAFT forward")
    p.add_argument("--video_glob", type=str, default="seed*.mp4")
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
def compute_flow(model, frames: torch.Tensor, device: str, batch: int) -> torch.Tensor:
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


def pairwise_distance(f_i: torch.Tensor, f_j: torch.Tensor, mask_fg: torch.Tensor,
                      eps: float = 1e-8) -> float:
    """Symmetric distance between two flow tensors [2, T, H, W] over a [H, W] foreground mask.

    Mirrors MotionModes/flowgen/motion_control/motion_losses_ours.py:991-1043
    (pairwise_difference_loss_angle_mag) but:
      - takes two flows directly (not flow-vs-list)
      - returns the raw avg_combined_diff (larger = more different),
        i.e. *not* the inverse-loss form
      - skips the `flow*2 - 1` rescale from L1006 because RAFT outputs are
        pixel displacements, not [0,1]-normalized.
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

    # Aggregate over foreground pixels of the mask, all frames
    fg = mask_fg.unsqueeze(0).expand_as(combined)
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

    # Load mask, resize to video resolution
    mask_fg = load_and_resize_mask(Path(args.mask_path), H, W)
    fg_count = int(mask_fg.sum())
    print(f"Mask foreground pixels: {fg_count}/{H*W} ({fg_count/(H*W):.2%})")
    if fg_count == 0:
        sys.exit("Mask has no foreground pixels after resize.")
    Image.fromarray((mask_fg.numpy().astype(np.uint8) * 255)).save(out_dir / "mask_used.png")

    # Load RAFT
    print("Loading RAFT-large...")
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=True).to(args.device).eval()

    # Compute flows (cache-aware)
    flows = []  # list of [2, T-1, H, W] CPU float32
    for seed, path in tqdm(videos, desc="flow"):
        cache_file = cache_dir / f"seed{seed}.pt" if cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            flow = torch.load(cache_file, map_location="cpu", weights_only=True)
        else:
            frames = load_video_frames(path)
            flow = compute_flow(model, frames, args.device, args.flow_batch)
            if cache_file is not None:
                torch.save(flow, cache_file)
            del frames
        flows.append(flow)

    # Free GPU model — distances are computed on flows directly
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Pairwise distance matrix on GPU
    N = len(flows)
    print(f"Computing {N}x{N} distance matrix...")
    device = torch.device(args.device)
    mask_fg_dev = mask_fg.to(device)
    flows_dev = [f.to(device) for f in flows]

    D = np.zeros((N, N), dtype=np.float64)
    for i in tqdm(range(N), desc="dist"):
        for j in range(i + 1, N):
            d = pairwise_distance(flows_dev[i], flows_dev[j], mask_fg_dev)
            D[i, j] = d
            D[j, i] = d

    np.save(out_dir / "distance_matrix.npy", D)
    print(f"Distance matrix: min={D[np.triu_indices(N, k=1)].min():.4f} "
          f"mean={D[np.triu_indices(N, k=1)].mean():.4f} "
          f"max={D[np.triu_indices(N, k=1)].max():.4f}")

    # Free flow tensors on GPU
    del flows_dev
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
