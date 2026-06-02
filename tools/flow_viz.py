"""
Render optical-flow color videos for quality check on a clip.

Supports canonical RAFT (torchvision; flow color via torchvision.utils.flow_to_image)
and MEMFOF (./memfof; flow color via memfof.utils.flow_viz.flow_to_image, matching
memfof/demo.py). Writes a flow-color mp4 (standard Middlebury color wheel: hue=direction,
saturation=magnitude).

Usage:
  python tools/flow_viz.py --model raft_large --video <mp4> --out <mp4>
  python tools/flow_viz.py --model memfof     --video <mp4> --out <mp4>
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent


class H264Writer:
    """Write RGB frames to an H.264/yuv420p mp4 via ffmpeg (universally playable;
    cv2's mp4v/MPEG-4-Part2 won't open in most browsers/players)."""

    def __init__(self, path, width, height, fps):
        import imageio_ffmpeg
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame_rgb):
        self.p.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def release(self):
        self.p.stdin.close()
        self.p.wait()


def load_frames_rgb(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames  # list of HWC uint8 RGB


@torch.inference_mode()
def run_raft(model_name, frames, device, writer, fps, W, H):
    from torchvision.models.optical_flow import (
        raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights)
    from torchvision.utils import flow_to_image
    if model_name == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False)
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False)
    model = model.eval().to(device)
    tfm = weights.transforms()
    for i in range(len(frames) - 1):
        a = torch.tensor(frames[i]).permute(2, 0, 1).unsqueeze(0)
        b = torch.tensor(frames[i + 1]).permute(2, 0, 1).unsqueeze(0)
        a, b = tfm(a, b)
        flow = model(a.to(device), b.to(device))[-1]  # [1,2,H,W]
        img = flow_to_image(flow[0]).cpu().permute(1, 2, 0).numpy()  # HWC uint8 RGB
        writer.write(img)


@torch.inference_mode()
def run_memfof(frames, device, writer, fps, W, H):
    sys.path.insert(0, str(_ROOT / "memfof"))
    from memfof.model import MEMFOF
    from memfof.utils.flow_viz import flow_to_image
    model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH").eval().to(device)
    rad_min = 0.02 * (H ** 2 + W ** 2) ** 0.5
    fmap_cache = [None] * 3
    window = [torch.tensor(frames[0]).permute(2, 0, 1).unsqueeze(0).float()]
    for f in frames:
        window.append(torch.tensor(f).permute(2, 0, 1).unsqueeze(0).float())
        if len(window) != 3:
            continue
        ft = torch.stack(window, dim=1).to(device)  # [1,3,C,H,W]
        out = model(ft, fmap_cache=fmap_cache)
        fwd = out["flow"][-1][:, 1]  # [1,2,H,W]
        img = flow_to_image(fwd.squeeze(0).permute(1, 2, 0).cpu().numpy(), rad_min=rad_min)
        writer.write(img)
        fmap_cache = out["fmap_cache"]; fmap_cache.pop(0); fmap_cache.append(None)
        window.pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["raft_large", "raft_small", "memfof"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="Upscale input frames by this factor before flow estimation "
                         "(dims rounded to multiples of 8). E.g. 2.0 -> 1280x768.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vpath = args.video if Path(args.video).is_absolute() else str(_ROOT / args.video)
    out = args.out if Path(args.out).is_absolute() else str(_ROOT / args.out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    frames = load_frames_rgb(vpath)
    H0, W0 = frames[0].shape[:2]
    if args.upscale != 1.0:
        W = int(round(W0 * args.upscale / 8) * 8)
        H = int(round(H0 * args.upscale / 8) * 8)
        frames = [cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC) for f in frames]
        print(f"Upscaled {W0}x{H0} -> {W}x{H} (x{args.upscale})")
    H, W = frames[0].shape[:2]
    print(f"Model={args.model}  Video={vpath}  {len(frames)} frames @ {W}x{H}  device={device}")

    writer = H264Writer(out, W, H, args.fps)
    if args.model.startswith("raft"):
        run_raft(args.model, frames, device, writer, args.fps, W, H)
    else:
        run_memfof(frames, device, writer, args.fps, W, H)
    writer.release()
    print(f"Wrote flow video: {out}")


if __name__ == "__main__":
    main()
