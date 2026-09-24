"""
make_demo.py - Turn the videos produced by main.py into a README-ready demo.

It takes the annotated camera-view video (and optionally the top-down pitch
video), trims a short clip, places the two views side by side, and exports:

    assets/demo.gif      small, autoplays in the README
    assets/demo.mp4      H.264 (plays in browsers / GitHub / YouTube)
    assets/preview.png   still frame, handy as a video thumbnail

Usage examples
--------------
    # side-by-side: camera view + top-down pitch
    python make_demo.py --annotated outputs/annotated.mp4 --pitch outputs/pitch.mp4

    # only the annotated video, 8 seconds starting at 12 s
    python make_demo.py --annotated outputs/annotated.mp4 --start 12 --duration 8

    # smaller GIF
    python make_demo.py --annotated outputs/annotated.mp4 --gif-width 560 --gif-fps 8

Adjust the input paths to whatever names your main.py writes into outputs/.

Requirements: opencv-python, numpy, imageio, imageio-ffmpeg
"""

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def read_clip(path, start, duration, fps_out):
    """Read `duration` seconds from `start`, subsampled to ~fps_out. Returns (frames, fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps_out))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * src_fps))

    frames = []
    for i in range(int(duration * src_fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {path} (check --start / --duration)")
    return frames, src_fps / step


def resize_to_height(frame, height):
    h, w = frame.shape[:2]
    new_w = int(round(w * height / h))
    return cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_AREA)


def make_even(frame):
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def scale_to_width(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return make_even(frame)
    new_h = int(round(h * width / w))
    return make_even(cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA))


def compose(annotated, pitch, height, max_width):
    n = min(len(annotated), len(pitch)) if pitch else len(annotated)
    out = []
    for i in range(n):
        left = resize_to_height(annotated[i], height)
        if pitch:
            right = resize_to_height(pitch[i], height)
            gap = np.full((height, 6, 3), 24, dtype=np.uint8)  # thin dark divider
            frame = np.hstack([left, gap, right])
        else:
            frame = left
        out.append(scale_to_width(frame, max_width))
    return out


def main():
    p = argparse.ArgumentParser(description="Create a README demo from pipeline output videos.")
    p.add_argument("--annotated", required=True, help="annotated camera-view video")
    p.add_argument("--pitch", default=None, help="top-down pitch video (optional)")
    p.add_argument("--out-dir", default="assets")
    p.add_argument("--start", type=float, default=0.0, help="clip start (seconds)")
    p.add_argument("--duration", type=float, default=8.0, help="clip length (seconds)")
    p.add_argument("--fps", type=int, default=20, help="fps of the MP4")
    p.add_argument("--height", type=int, default=420, help="height of each view before stacking")
    p.add_argument("--width", type=int, default=1280, help="max width of the MP4")
    p.add_argument("--crf", type=int, default=26, help="MP4 quality (lower = better/larger)")
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--gif-width", type=int, default=720, help="max width of the GIF")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-mp4", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading clips...")
    ann, fps = read_clip(args.annotated, args.start, args.duration, args.fps)
    pit = None
    if args.pitch:
        pit, _ = read_clip(args.pitch, args.start, args.duration, args.fps)

    frames = compose(ann, pit, args.height, args.width)
    print(f"Composed {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}, ~{fps:.1f} fps")

    # still preview
    preview = frames[len(frames) // 2]
    imageio.imwrite(out_dir / "preview.png", preview)
    print(f"Saved {out_dir / 'preview.png'}")

    # MP4 (H.264 so it plays in browsers)
    if not args.no_mp4:
        mp4_path = out_dir / "demo.mp4"
        with imageio.get_writer(
            mp4_path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=["-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ) as writer:
            for f in frames:
                writer.append_data(f)
        print(f"Saved {mp4_path} ({mp4_path.stat().st_size / 1e6:.1f} MB)")

    # GIF (autoplays in the README)
    if not args.no_gif:
        gif_step = max(1, round(fps / args.gif_fps))
        gif_frames = [scale_to_width(f, args.gif_width) for f in frames[::gif_step]]
        gif_path = out_dir / "demo.gif"
        imageio.mimsave(gif_path, gif_frames, duration=1000 * gif_step / fps, loop=0)
        size_mb = gif_path.stat().st_size / 1e6
        print(f"Saved {gif_path} ({size_mb:.1f} MB)")
        if size_mb > 10:
            print("GIF is over 10 MB. Try --duration 5, --gif-width 560 or --gif-fps 8.")


if __name__ == "__main__":
    main()
