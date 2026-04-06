"""
Visualize the motion-diversity space of N videos and the K selected ones.

Produces 2 figures in --output_dir:

1. cluster_map.png
   - 2D MDS embedding of the N videos from the precomputed distance matrix
     (the same matrix used to pick the diverse subset)
   - K-means clustering on the embedding (color = cluster)
   - Selected seeds highlighted as red stars with seed labels

2. motion_directions.png
   - Per-selected-video flow visualization, side-by-side
   - Each panel: first frame of the video + a big arrow drawn from the
     mask centroid representing the *mean flow vector inside the mask*
     (averaged over all frame pairs and all foreground pixels)
   - Includes the seed and the arrow magnitude in the title

Inputs are read from the same paths used by tools/select_diverse_videos.py
so this is a pure post-processing visualization step.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from video_reader import PyVideoReader

SEED_RE = re.compile(r"seed(\d+)_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", type=str, default="output_helios/helios-distilled_i2v")
    p.add_argument("--mask_path", type=str, default="output_helios/masks/first_frame_ff42_mask.png")
    p.add_argument("--flow_cache_dir", type=str, default="output_helios/helios-distilled_i2v_flows")
    p.add_argument("--selection_dir", type=str, default="output_helios/helios-distilled_i2v_diverse_top4",
                   help="Directory containing distance_matrix.npy and selected.json")
    p.add_argument("--output_dir", type=str, default="output_helios/helios-distilled_i2v_diverse_top4")
    p.add_argument("--num_clusters", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def find_videos(video_dir: Path):
    out = []
    for p in sorted(video_dir.glob("seed*.mp4")):
        m = SEED_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_first_frame(path: Path) -> np.ndarray:
    vr = PyVideoReader(str(path), threads=0)
    buf = vr.decode()
    del vr
    return buf[0]  # [H, W, 3] uint8


def load_resized_mask(mask_path: Path, h: int, w: int) -> np.ndarray:
    img = Image.open(mask_path).convert("L")
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)
    return np.array(img) > 127  # [H, W] bool


def plot_cluster_map(D, seeds, selected_seeds, num_clusters, seed_random, out_path):
    """MDS embedding + KMeans + selected highlights."""
    print(f"Computing MDS embedding for {len(seeds)} videos...")
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed_random,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(D)  # [N, 2]

    print(f"KMeans clustering into {num_clusters} clusters...")
    km = KMeans(n_clusters=num_clusters, random_state=seed_random, n_init=10)
    labels = km.fit_predict(coords)

    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.get_cmap("tab10")
    for c in range(num_clusters):
        m = labels == c
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            s=120,
            c=[cmap(c)],
            alpha=0.65,
            edgecolors="black",
            linewidths=0.6,
            label=f"cluster {c}",
        )

    # seed labels (small)
    for (x, y), s in zip(coords, seeds):
        ax.text(x, y, str(s), fontsize=6.5, ha="center", va="center", color="black")

    # selected highlights
    selected_mask = np.array([s in selected_seeds for s in seeds])
    ax.scatter(
        coords[selected_mask, 0],
        coords[selected_mask, 1],
        s=600,
        marker="*",
        facecolors="none",
        edgecolors="red",
        linewidths=2.5,
        label="selected",
        zorder=5,
    )

    ax.set_title(
        f"Motion-diversity MDS embedding (N={len(seeds)})\n"
        f"distance = masked-region flow magnitude+angle (motionmodes kernel)"
    )
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return coords, labels


def plot_motion_directions(videos_by_seed, mask_fg, flow_cache_dir, selected_seeds, out_path):
    """Per-selected-video first frame with mean-flow arrow drawn over the mask."""
    n = len(selected_seeds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    H, W = mask_fg.shape
    fg_yx = np.stack(np.where(mask_fg), axis=1)  # [P, 2] in (y, x)
    centroid = fg_yx.mean(axis=0)  # (cy, cx)

    for ax, seed in zip(axes, selected_seeds):
        path = videos_by_seed[seed]
        first = load_first_frame(path)  # [H, W, 3] uint8

        # load cached flow [2, T-1, H, W]
        flow_path = Path(flow_cache_dir) / f"seed{seed}.pt"
        flow = torch.load(flow_path, map_location="cpu", weights_only=True).numpy()
        # mean over time and over mask foreground pixels
        # flow[0] = u (x-component), flow[1] = v (y-component)
        u = flow[0][:, mask_fg].mean()
        v = flow[1][:, mask_fg].mean()
        mag = float(np.hypot(u, v))
        ang = float(np.degrees(np.arctan2(v, u)))

        # draw
        ax.imshow(first)
        # mask outline
        ax.contour(mask_fg.astype(float), levels=[0.5], colors="cyan", linewidths=1.5)

        # arrow scaled so it's visible regardless of magnitude
        max_extent = 0.35 * min(H, W)
        scale = max_extent / max(mag, 1e-3)
        arr = FancyArrowPatch(
            (centroid[1], centroid[0]),
            (centroid[1] + u * scale, centroid[0] + v * scale),
            arrowstyle="->",
            mutation_scale=25,
            color="red",
            linewidth=3,
        )
        ax.add_patch(arr)

        ax.set_title(f"seed {seed}\nmean flow: |v|={mag:.2f} px, angle={ang:+.0f}°")
        ax.axis("off")

    fig.suptitle(
        "Mean optical flow direction inside the mask (cat region)\n"
        "averaged over all frame pairs and all foreground pixels",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_flow_color_per_pixel(videos_by_seed, mask_fg, flow_cache_dir, selected_seeds, out_path):
    """Show per-pixel mean flow as a color-coded HSV image (hue=direction, sat=mag) over the first frame, masked."""
    from matplotlib.colors import hsv_to_rgb

    n = len(selected_seeds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    H, W = mask_fg.shape

    # find global magnitude scale across selected for consistent saturation
    mags = []
    flows = {}
    for seed in selected_seeds:
        f = torch.load(Path(flow_cache_dir) / f"seed{seed}.pt", map_location="cpu", weights_only=True).numpy()
        mean_flow = f.mean(axis=1)  # [2, H, W] mean over time
        flows[seed] = mean_flow
        mag = np.hypot(mean_flow[0], mean_flow[1])
        mags.append(mag[mask_fg].max())
    global_max_mag = max(mags) if mags else 1.0

    for ax, seed in zip(axes, selected_seeds):
        first = load_first_frame(videos_by_seed[seed])
        mean_flow = flows[seed]
        u, v = mean_flow[0], mean_flow[1]

        ang = (np.arctan2(v, u) + np.pi) / (2 * np.pi)  # [0,1]
        mag = np.hypot(u, v) / max(global_max_mag, 1e-6)
        mag = np.clip(mag, 0, 1)

        hsv = np.stack([ang, mag, np.ones_like(ang)], axis=-1)
        rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
        # composite: dim background, mask region shows flow color
        composite = first.astype(np.float32) * 0.35
        m3 = mask_fg[..., None]
        composite = np.where(m3, rgb.astype(np.float32), composite)
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        ax.imshow(composite)
        ax.contour(mask_fg.astype(float), levels=[0.5], colors="white", linewidths=1.0)
        ax.set_title(f"seed {seed}")
        ax.axis("off")

    fig.suptitle(
        "Per-pixel time-averaged flow inside mask\n"
        "hue = direction, brightness = magnitude (global-normalized)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads((Path(args.selection_dir) / "selected.json").read_text())
    selected_seeds = selection["selected_seeds"]
    print(f"Selected seeds: {selected_seeds}")

    D = np.load(Path(args.selection_dir) / "distance_matrix.npy")
    print(f"Distance matrix shape: {D.shape}")

    videos = find_videos(Path(args.video_dir))
    seeds = [s for s, _ in videos]
    videos_by_seed = {s: p for s, p in videos}

    # mask
    H, W = load_first_frame(videos[0][1]).shape[:2]
    mask_fg = load_resized_mask(Path(args.mask_path), H, W)
    print(f"Mask FG pixels: {mask_fg.sum()}/{H*W}")

    # 1. cluster map
    plot_cluster_map(
        D=D,
        seeds=seeds,
        selected_seeds=selected_seeds,
        num_clusters=args.num_clusters,
        seed_random=args.seed,
        out_path=out_dir / "cluster_map.png",
    )

    # 2. mean flow direction arrows
    plot_motion_directions(
        videos_by_seed=videos_by_seed,
        mask_fg=mask_fg,
        flow_cache_dir=args.flow_cache_dir,
        selected_seeds=selected_seeds,
        out_path=out_dir / "motion_directions.png",
    )

    # 3. per-pixel flow color
    plot_flow_color_per_pixel(
        videos_by_seed=videos_by_seed,
        mask_fg=mask_fg,
        flow_cache_dir=args.flow_cache_dir,
        selected_seeds=selected_seeds,
        out_path=out_dir / "motion_flow_color.png",
    )


if __name__ == "__main__":
    main()
