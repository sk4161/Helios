"""
Mask an optical-flow color video with a SAM2 foreground mask so only the tracked
object's flow remains (background flow noise is removed). CPU-only compositing of
an existing flow mp4 + a saved bool mask tensor [T,H,W]. Writes H.264.

Usage:
  python tools/mask_flow.py --flow_video <flow.mp4> --mask_pt <mask.pt> --out <out.mp4> [--bg black|white]
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import imageio_ffmpeg


def read_frames_bgr(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)  # BGR
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_video", required=True)
    ap.add_argument("--mask_pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg", choices=["black", "white"], default="black",
                    help="Background fill where mask is False.")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--mask_offset", type=int, default=0,
                    help="mask index = flow index + offset (alignment nudge).")
    args = ap.parse_args()

    flow = read_frames_bgr(args.flow_video)        # list BGR, length ~T-1
    mask = torch.load(args.mask_pt, map_location="cpu", weights_only=True).numpy().astype(bool)  # [T,H,W]
    Tf, (H, W) = len(flow), flow[0].shape[:2]
    Tm = mask.shape[0]
    print(f"flow frames={Tf} @ {W}x{H} ; mask frames={Tm} @ {mask.shape[2]}x{mask.shape[1]}")

    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    fill = 0 if args.bg == "black" else 255
    kept = 0
    for i in range(Tf):
        mi = min(max(i + args.mask_offset, 0), Tm - 1)
        m = mask[mi]
        if (m.shape[0] != H) or (m.shape[1] != W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        frame = flow[i].copy()
        frame[~m] = fill
        kept += m.mean()
        proc.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Wrote {out}  (mean kept fg fraction {kept/Tf:.4f})")


if __name__ == "__main__":
    main()
