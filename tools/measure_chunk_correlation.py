"""
Phase A of the cascade-pruning experiment.

Simulates a 64 -> 32 -> 16 -> 8 -> 4 cascade using the **temporal slice** of
the existing per-video flow + tracked-mask caches. Compares the cascade's
final 4 against the post-hoc gold-standard top-4 (computed on the full 97
frames). Reports per-stage metrics.

This answers the question:
  "If we had only L frames of motion to look at, can we still make the
   right pruning decision at this stage?"

It does NOT answer the related (and more relevant) question of whether
half-denoised chunk1 outputs are informative — that needs Phase B
(actual partial inference). See plan file.

Reuses pairwise_distance() and farthest_point_sampling() from
tools/select_diverse_videos.py to keep distance semantics identical.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

# Reuse the kernel from select_diverse_videos.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_diverse_videos import pairwise_distance, farthest_point_sampling  # noqa: E402

SEED_RE = re.compile(r"seed(\d+)")


def parse_schedule(s: str) -> list[tuple[int, int]]:
    """'16:32,33:16,66:8,97:4' -> [(16,32),(33,16),(66,8),(97,4)]"""
    out = []
    for part in s.split(","):
        L, K = part.strip().split(":")
        out.append((int(L), int(K)))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flow_cache_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, required=True)
    p.add_argument("--full_distance", type=str, required=True,
                   help="Path to the gold-standard distance_matrix.npy")
    p.add_argument("--full_selected", type=str, required=True,
                   help="Path to the gold-standard selected.json")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--schedule", type=str, default="16:32,33:16,66:8,97:4",
                   help="Comma-separated 'L:K' stages. L = number of video frames "
                        "observed at this stage; K = number of seeds kept after this stage.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_caches(flow_dir: Path, mask_dir: Path, device: str):
    """Discover seeds present in BOTH flow and mask caches, return sorted lists."""
    flow_seeds = {int(m.group(1)): p for p in flow_dir.glob("seed*.pt")
                  for m in [SEED_RE.search(p.name)] if m}
    mask_seeds = {int(m.group(1)): p for p in mask_dir.glob("seed*.pt")
                  for m in [SEED_RE.search(p.name)] if m}
    common = sorted(set(flow_seeds) & set(mask_seeds))
    if not common:
        sys.exit(f"No overlap between flow_cache_dir ({len(flow_seeds)} files) "
                 f"and mask_dir ({len(mask_seeds)} files)")
    print(f"Loading {len(common)} seeds from {flow_dir} & {mask_dir}")
    flows = []
    masks = []
    for s in tqdm(common, desc="load"):
        flows.append(torch.load(flow_seeds[s], map_location=device, weights_only=True))
        masks.append(torch.load(mask_seeds[s], map_location=device, weights_only=True))
    return common, flows, masks


def slice_to_length(flows: list[torch.Tensor], masks: list[torch.Tensor], L: int):
    """Return flow [2, L-1, H, W] and mask [L, H, W] views for each video.

    L is the number of *video* frames observed; the corresponding number of
    consecutive flow pairs is L-1.
    """
    out_flows = []
    out_masks = []
    for f, m in zip(flows, masks):
        # f: [2, T_flow, H, W], m: [T_mask, H, W]
        T_flow = f.shape[1]
        T_mask = m.shape[0]
        # The mask is the per-frame video mask; flows are pair-aligned.
        # Cap L by what's actually available.
        L_eff = min(L, T_mask, T_flow + 1)
        out_flows.append(f[:, : L_eff - 1].contiguous())
        out_masks.append(m[:L_eff].contiguous())
    return out_flows, out_masks


def compute_distance_subset(flows, masks, indices: list[int]) -> np.ndarray:
    """Pairwise distance matrix over the given subset of seed indices.
    Result is K x K (K = len(indices)) where row/col i corresponds to indices[i]."""
    K = len(indices)
    D = np.zeros((K, K), dtype=np.float64)
    for ii in range(K):
        for jj in range(ii + 1, K):
            i, j = indices[ii], indices[jj]
            d = pairwise_distance(flows[i], flows[j], masks[i] & masks[j])
            D[ii, jj] = d
            D[jj, ii] = d
    return D


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = parse_schedule(args.schedule)
    print(f"Schedule (L, K_out): {schedule}")
    if schedule[-1][1] < 1:
        sys.exit("Final K_out must be >= 1")

    # Load gold standard
    D_full = np.load(args.full_distance)
    selected = json.loads(Path(args.full_selected).read_text())
    post_hoc_top = selected["selected_seeds"]
    K_final = schedule[-1][1]
    print(f"Gold-standard top-{len(post_hoc_top)}: {post_hoc_top}")

    # Load caches
    seeds, flows, masks = load_caches(Path(args.flow_cache_dir), Path(args.mask_dir), args.device)
    if len(seeds) != D_full.shape[0]:
        print(f"WARN: cache has {len(seeds)} seeds but D_full is {D_full.shape}; "
              f"assuming row i = seed seeds[i]")
    seed_to_idx = {s: i for i, s in enumerate(seeds)}

    # Cascade loop ----------------------------------------------------------
    survivors_idx = list(range(len(seeds)))  # indices into `seeds` (and flows/masks)
    stages_report = []
    for stage_no, (L, K_out) in enumerate(schedule, start=1):
        print(f"\n=== stage {stage_no}: L={L} frames, {len(survivors_idx)} -> {K_out} ===")
        # Slice flow + mask to first L video frames
        sliced_flows, sliced_masks = slice_to_length(flows, masks, L)
        # Distance matrix over current survivors only
        D_L = compute_distance_subset(sliced_flows, sliced_masks, survivors_idx)
        # FPS to keep K_out
        keep_local = farthest_point_sampling(D_L, K_out)
        new_survivors_idx = [survivors_idx[k] for k in keep_local]

        # Spearman correlation between D_L and the corresponding submatrix of D_full
        D_full_sub = D_full[np.ix_(survivors_idx, survivors_idx)]
        if D_L.shape[0] > 2:
            iu = np.triu_indices(D_L.shape[0], k=1)
            rho, _ = spearmanr(D_L[iu], D_full_sub[iu])
        else:
            rho = float("nan")

        # Survival of post-hoc top-4 within this stage's input survivors
        post_hoc_idx = {seed_to_idx[s] for s in post_hoc_top}
        post_hoc_in_input = post_hoc_idx & set(survivors_idx)
        post_hoc_in_output = post_hoc_idx & set(new_survivors_idx)

        stage_record = {
            "stage": stage_no,
            "L_frames": L,
            "K_in": len(survivors_idx),
            "K_out": K_out,
            "spearman_with_full": float(rho),
            "post_hoc_top_in_input": sorted([seeds[i] for i in post_hoc_in_input]),
            "post_hoc_top_in_output": sorted([seeds[i] for i in post_hoc_in_output]),
            "post_hoc_top_survived": len(post_hoc_in_output),
            "post_hoc_top_lost_this_stage": sorted(
                [seeds[i] for i in (post_hoc_in_input - post_hoc_in_output)]
            ),
            "survivor_seeds": sorted([seeds[i] for i in new_survivors_idx]),
        }
        stages_report.append(stage_record)
        print(f"  spearman vs full: {rho:.4f}")
        print(f"  post-hoc top survived: {len(post_hoc_in_output)}/{len(post_hoc_top)} "
              f"(in_input was {len(post_hoc_in_input)})")
        if stage_record["post_hoc_top_lost_this_stage"]:
            print(f"  ✗ lost: {stage_record['post_hoc_top_lost_this_stage']}")

        survivors_idx = new_survivors_idx

    cascade_top = sorted([seeds[i] for i in survivors_idx])
    final_overlap = len(set(cascade_top) & set(post_hoc_top))

    # Quality: min/mean pair distance under D_full for both sets
    def stats(idx_list):
        if len(idx_list) < 2:
            return {"min": None, "mean": None}
        sub = D_full[np.ix_(idx_list, idx_list)]
        iu = np.triu_indices(len(idx_list), k=1)
        return {"min": float(sub[iu].min()), "mean": float(sub[iu].mean())}

    cascade_idx = [seed_to_idx[s] for s in cascade_top]
    post_hoc_idx = [seed_to_idx[s] for s in post_hoc_top]
    cascade_stats = stats(cascade_idx)
    post_hoc_stats = stats(post_hoc_idx)

    report = {
        "schedule": schedule,
        "n_videos_total": len(seeds),
        "post_hoc_top": post_hoc_top,
        "cascade_top": cascade_top,
        "final_overlap": final_overlap,
        "post_hoc_min_pair_dist": post_hoc_stats["min"],
        "post_hoc_mean_pair_dist": post_hoc_stats["mean"],
        "cascade_min_pair_dist": cascade_stats["min"],
        "cascade_mean_pair_dist": cascade_stats["mean"],
        "stages": stages_report,
    }
    out_report = out_dir / "cascade_report.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"\n=== Final ===")
    print(f"post-hoc top: {post_hoc_top}")
    print(f"cascade  top: {cascade_top}")
    print(f"overlap:      {final_overlap}/{len(post_hoc_top)}")
    print(f"min pair dist: post_hoc={post_hoc_stats['min']:.4f} "
          f"cascade={cascade_stats['min']:.4f}")
    print(f"Wrote {out_report}")

    # Diagram: stage-by-stage survivors -------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 0.18 * len(seeds)), 4 + len(schedule) * 0.4))
    n = len(seeds)
    x = np.arange(n)
    # rows: stage 0 (all 64) at top, then each cascade stage
    rows = [list(range(n))] + [[seed_to_idx[s] for s in st["survivor_seeds"]] for st in stages_report]
    row_labels = [f"start (n={n})"] + [
        f"stage {st['stage']}: L={st['L_frames']}, n={len(st['survivor_seeds'])}"
        for st in stages_report
    ]
    post_hoc_idx_set = {seed_to_idx[s] for s in post_hoc_top}
    for r, (alive_idxs, label) in enumerate(zip(rows, row_labels)):
        alive_set = set(alive_idxs)
        for i in range(n):
            if i in alive_set:
                color = "tab:red" if i in post_hoc_idx_set else "tab:blue"
            else:
                color = "lightgray"
            ax.add_patch(plt.Rectangle((i - 0.45, -r - 0.45), 0.9, 0.9,
                                       facecolor=color, edgecolor="white", linewidth=0.5))
        ax.text(-1, -r, label, ha="right", va="center", fontsize=9)
    ax.set_xlim(-12, n)
    ax.set_ylim(-len(rows) - 0.5, 0.5)
    ax.set_xticks(x[::4])
    ax.set_xticklabels([str(seeds[i]) for i in x[::4]], fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("seed (every 4th label)")
    ax.set_title(
        f"Cascade survivors ({args.schedule})\n"
        f"red = post-hoc gold top-{len(post_hoc_top)}, blue = other survivors, gray = pruned"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cascade_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir/'cascade_diagram.png'}")


if __name__ == "__main__":
    main()
