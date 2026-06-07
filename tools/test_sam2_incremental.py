"""Isolate-test incremental SAM2 vs full re-track on an existing clip.

Tracks a real fish_i2v mp4 two ways and reports per-frame IoU:
  (A) full:        sam2_init on all T frames
  (B) incremental: sam2_init on [:33] then sam2_extend on [33:65], [65:97]
If incremental memory continuity works, IoU should be ~1.0 on every frame.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from video_reader import PyVideoReader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))
import progressive_pruning as pp  # reuse sam2_init / sam2_extend / preprocessing
from sam2_common import best_box, detect_boxes_with_gdino  # noqa: E402


def iou(a, b):
    a = a.bool(); b = b.bool()
    inter = (a & b).sum().item(); uni = (a | b).sum().item()
    return 1.0 if uni == 0 else inter / uni


def main():
    import os
    device = "cuda"
    mp4 = sorted(glob.glob(str(_ROOT / "output_helios/fish_i2v/seed11_*.mp4")))[0]
    buf = PyVideoReader(mp4, threads=0).decode()  # [T,H,W,3] uint8
    T = min(97, buf.shape[0])
    frames = torch.from_numpy(buf[:T].copy()).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]
    print(f"clip={mp4} T={T} HxW={frames.shape[-2]}x{frames.shape[-1]}")

    from PIL import Image
    img = Image.fromarray(buf[0])
    bbox = best_box(*detect_boxes_with_gdino(img, "a fish.",
                                             "IDEA-Research/grounding-dino-tiny", device)[:2])
    print("bbox", bbox)

    # build predictor (same shadow-fix dance as the orchestrator)
    saved_path, saved_cwd = list(sys.path), os.getcwd()
    os.chdir("/tmp"); sys.path[:] = [p for p in sys.path if p not in ("", str(_ROOT))]
    pp.patch_vos_compile_mode()
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(_ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt"), device=device, vos_optimized=True)
    sys.path[:] = saved_path; os.chdir(saved_cwd)

    tmp = Path("/tmp/sam2test"); tmp.mkdir(exist_ok=True)

    # (A) full
    st_full, m_full = pp.sam2_init(predictor, frames, bbox, str(tmp / "full.mp4"))
    predictor.reset_state(st_full)

    # (B) incremental 33 / 32 / 32
    st, m0 = pp.sam2_init(predictor, frames[:33], bbox, str(tmp / "inc.mp4"))
    m1 = pp.sam2_extend(predictor, st, frames[33:65])
    m2 = pp.sam2_extend(predictor, st, frames[65:97])
    m_inc = torch.cat([m0, m1, m2], dim=0)
    predictor.reset_state(st)

    print(f"full {tuple(m_full.shape)}  inc {tuple(m_inc.shape)}")
    ious = [iou(m_full[t], m_inc[t]) for t in range(T)]
    print("per-frame IoU (every 8):", [round(ious[t], 3) for t in range(0, T, 8)])
    print(f"mean IoU={np.mean(ious):.4f}  min IoU={np.min(ious):.4f}  "
          f"frames<0.9: {sum(1 for x in ious if x < 0.9)}/{T}")
    print(f"fg_frac full={m_full.float().mean():.4f} inc={m_inc.float().mean():.4f}")


if __name__ == "__main__":
    main()
