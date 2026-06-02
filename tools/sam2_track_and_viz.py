"""
Run native SAM2 video tracking on one clip and write a QUALITY-CHECK overlay mp4
(semi-transparent mask + contour over the original frames). Seeds the first frame
with GroundingDINO (text prompt) unless --bbox is given. Also saves the raw bool
mask tensor [T,H,W] as <out_prefix>_mask.pt.

Usage:
  python tools/sam2_track_and_viz.py --video <mp4> --text "a fish." --out_prefix <path>
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HELIOS_ROOT = Path(__file__).resolve().parent.parent

# sam2 parent-dir shadow workaround (must precede `import sam2`).
os.chdir("/tmp")
if sys.path and (sys.path[0] == "" or Path(sys.path[0]).resolve() == _HELIOS_ROOT):
    sys.path = sys.path[1:]


def detect_bbox(image_rgb, text, device, box_thr=0.35, text_thr=0.25):
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
    from PIL import Image
    img = Image.fromarray(image_rgb)
    proc = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
    inputs = proc(images=img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, threshold=box_thr, text_threshold=text_thr, target_sizes=[img.size[::-1]])[0]
    boxes, scores = res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO found no boxes for {text!r}")
    best = int(np.argmax(scores))
    print(f"GroundingDINO bbox: {boxes[best].tolist()} (score={scores[best]:.3f})")
    del model, proc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return boxes[best].tolist()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", default="a fish.")
    ap.add_argument("--bbox", type=float, nargs=4, default=None)
    ap.add_argument("--out_prefix", required=True, help="Output path prefix (no extension).")
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam2_ckpt", default=str(_HELIOS_ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vpath = args.video if Path(args.video).is_absolute() else str(_HELIOS_ROOT / args.video)
    out_prefix = args.out_prefix if Path(args.out_prefix).is_absolute() else str(_HELIOS_ROOT / args.out_prefix)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Read all frames (RGB uint8) for seeding + overlay.
    from video_reader import PyVideoReader
    buf = PyVideoReader(vpath, threads=0).decode()  # [T,H,W,3] RGB uint8
    T, H, W = buf.shape[:3]
    print(f"Video: {vpath}  {T} frames @ {W}x{H}")

    bbox = list(args.bbox) if args.bbox else detect_bbox(buf[0], args.text, device)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device, vos_optimized=False)

    state = predictor.init_state(video_path=vpath)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                    box=np.array(bbox, dtype=np.float32))
    masks = torch.zeros((T, H, W), dtype=torch.bool)
    for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        masks[fidx] = (mask_logits[0, 0] > 0.0).cpu()

    mask_np = masks.numpy()
    fg = mask_np.mean()
    print(f"Tracked. mean foreground fraction = {fg:.4f}")
    torch.save(masks, f"{out_prefix}_mask.pt")

    # Overlay: green tint (50%) inside mask + yellow contour, draw seed bbox on frame 0.
    # H.264/yuv420p via ffmpeg (cv2's mp4v won't open in most browsers/players).
    import subprocess, imageio_ffmpeg
    out_mp4 = f"{out_prefix}_overlay.mp4"
    _cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    proc = subprocess.Popen(_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    color = np.array([0, 255, 0], dtype=np.uint8)
    for i in range(T):
        rgb = buf[i].copy()
        m = mask_np[i]
        if m.any():
            rgb[m] = (0.5 * rgb[m] + 0.5 * color).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, cnts, -1, (255, 255, 0), 2)
        if i == 0:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Wrote overlay: {out_mp4}")
    print(f"Wrote mask:    {out_prefix}_mask.pt")


if __name__ == "__main__":
    main()
