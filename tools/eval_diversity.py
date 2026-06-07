"""Quantitative DIVERSITY evaluation of 4-video sets (random vs each prune schedule).

Two metrics, mean over all unordered pairs within a set (larger = more diverse):
  - visual (CLIP): 1 - cos(mean-frame CLIP embedding_i, _j)   [independent of selection]
  - motion (RAFT): pairwise_distance(flow_i, flow_j)          [the QIP selection objective]

For the `random` reference we evaluate a LARGER pool (e.g. 16) so the mean-pairwise value
robustly estimates the expected diversity of an unselected pair (a single random 4-set is
noisy). Pruned sets are exactly the 4 survivors. Both report mean pairwise distance, so the
numbers are directly comparable.
"""
import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_ROOT / ".torchinductor_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from video_reader import PyVideoReader  # noqa: E402

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large  # noqa: E402
from select_diverse_videos import compute_flow_raft, pairwise_distance  # noqa: E402
from progressive_pruning import clip_embed, _clip_tensor  # noqa: E402


def load_frames(path):
    buf = PyVideoReader(str(path), threads=0).decode()       # [T,H,W,3] uint8
    return torch.from_numpy(buf).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], metavar="NAME=GLOB",
                    help="repeatable; e.g. random=output_helios/gen16_random/*.mp4")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip_frames", type=int, default=8)
    ap.add_argument("--flow_hw", type=int, nargs=2, default=[96, 160],
                    help="downsample flow to this HxW for the pairwise metric (relative; cheap).")
    ap.add_argument("--out", default="output_helios/diversity_eval.json")
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).to(device).eval()
    from transformers import CLIPModel, CLIPProcessor
    m_clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    text_emb = torch.zeros(1, 1, device=device)  # unused; clip_embed needs a text_emb arg only for signature

    results = {}
    for spec in args.set:
        name, gpat = spec.split("=", 1)
        paths = sorted(glob.glob(gpat if os.path.isabs(gpat) else str(_ROOT / gpat)))
        if len(paths) < 2:
            print(f"[skip] {name}: <2 videos ({gpat})"); continue
        print(f"\n=== {name}: {len(paths)} videos ===")

        clip_vecs, flows = [], []
        with torch.inference_mode():
            for p in paths:
                fr = load_frames(p)                                   # [T,3,H,W] uint8
                T = fr.shape[0]
                # CLIP: mean over uniformly-sampled frames, then L2-normalize
                idx = torch.linspace(0, T - 1, min(args.clip_frames, T)).round().long().tolist()
                emb = clip_embed(fr[idx], m_clip, text_emb, CLIP_MEAN, CLIP_STD, device)  # [k,D]
                clip_vecs.append(F.normalize(emb.mean(0, keepdim=True), dim=-1))           # [1,D]
                # motion: RAFT flow, downsample spatially for a cheap relative pairwise metric
                fl = compute_flow_raft(raft, fr, device, batch=8, num_flow_updates=8,
                                       compute_dtype=torch.bfloat16)  # [2,T-1,H,W]
                fl = F.interpolate(fl, size=tuple(args.flow_hw), mode="bilinear", align_corners=False)
                flows.append(fl.float())

        names = [Path(p).stem for p in paths]
        vis_pairs, mot_pairs = [], []
        for i, j in itertools.combinations(range(len(paths)), 2):
            vis = 1.0 - float((clip_vecs[i] @ clip_vecs[j].T).item())
            mot = pairwise_distance(flows[i], flows[j])
            vis_pairs.append(vis); mot_pairs.append(mot)

        results[name] = {
            "n": len(paths), "videos": names,
            "visual_div_mean": round(float(np.mean(vis_pairs)), 4),
            "visual_div_std": round(float(np.std(vis_pairs)), 4),
            "motion_div_mean": round(float(np.mean(mot_pairs)), 4),
            "motion_div_std": round(float(np.std(mot_pairs)), 4),
            "n_pairs": len(vis_pairs),
        }
        r = results[name]
        print(f"  visual_div = {r['visual_div_mean']} ± {r['visual_div_std']}   "
              f"motion_div = {r['motion_div_mean']} ± {r['motion_div_std']}   ({r['n_pairs']} pairs)")

    Path(args.out if os.path.isabs(args.out) else str(_ROOT / args.out)).write_text(
        json.dumps(results, indent=2))
    print("\n=== SUMMARY (mean pairwise, larger = more diverse) ===")
    print(f"{'set':<14} {'visual_div':>12} {'motion_div':>12}")
    for name, r in results.items():
        print(f"{name:<14} {r['visual_div_mean']:>12} {r['motion_div_mean']:>12}")


if __name__ == "__main__":
    main()
