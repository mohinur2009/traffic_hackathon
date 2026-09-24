"""Run detector + tracker on videos and save ("cache") the tracks.

One row per (frame, object):
    frame, t_sec, track_id, cls, cls_name, conf, x1, y1, x2, y2, cx, cy
(cx, cy) = bottom-centre of the box = where the object touches the road.

Usage:
    python src/track.py --videos samples/ --out cache/ --stride 2
    python src/track.py --video samples/a.mp4 --out cache/ --max-sec 60
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# COCO class ids we care about (road users only)
KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLUMNS = ["frame", "t_sec", "track_id", "cls", "cls_name", "conf", "x1", "y1", "x2", "y2"]


def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def track_video(video_path, weights="yolo11s.pt", tracker="bytetrack.yaml",
                stride=2, imgsz=960, conf=0.25, max_sec=None):
    """Detect + track one video. Returns (tracks DataFrame, meta dict)."""
    set_seeds(0)
    model = YOLO(weights)  # new model per video = clean tracker state
    half = torch.cuda.is_available()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = n_frames if max_sec is None else min(n_frames, int(max_sec * fps))

    rows, idx = [], 0
    t0 = time.perf_counter()
    while idx < last:
        if idx % stride:                      # skipped frame: grab is cheaper than read
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker=tracker, classes=list(KEEP),
                          conf=conf, imgsz=imgsz, half=half, verbose=False)[0]
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            cls = boxes.cls.int().cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, c, s in zip(xyxy, ids, cls, scores):
                rows.append((idx, idx / fps, int(tid), int(c), KEEP[int(c)], float(s),
                             float(x1), float(y1), float(x2), float(y2)))
        idx += 1
    elapsed = time.perf_counter() - t0
    cap.release()

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["cy"] = df["y2"]

    processed_sec = idx / fps
    meta = {
        "video": Path(video_path).name, "fps": fps, "width": width, "height": height,
        "n_frames": n_frames, "duration_sec": n_frames / fps,
        "processed_sec": processed_sec, "weights": weights, "tracker": tracker,
        "stride": stride, "imgsz": imgsz, "conf": conf,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "elapsed_sec": elapsed,
        # < 1.0 means faster than real time. Budget for Part A + Part B together is 3.0
        "speed_ratio": elapsed / processed_sec if processed_sec else None,
        **track_quality(df, fps),
    }
    return df, meta


def track_quality(df: pd.DataFrame, fps: float) -> dict:
    """Simple health numbers. Many very short tracks = the tracker keeps losing objects."""
    if df.empty:
        return {"n_tracks": 0, "median_track_sec": 0.0, "short_track_share": 0.0}
    lengths = df.groupby("track_id")["t_sec"].agg(lambda t: t.max() - t.min())
    return {
        "n_tracks": int(lengths.size),
        "median_track_sec": float(lengths.median()),
        "short_track_share": float((lengths < 1.0).mean()),  # tracks shorter than 1 s
    }


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--videos", type=Path, help="folder with .mp4 files")
    ap.add_argument("--out", type=Path, default=Path("cache"))
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-sec", type=float, default=None)
    args = ap.parse_args()

    videos = [args.video] if args.video else sorted(args.videos.glob("*.mp4"))
    args.out.mkdir(parents=True, exist_ok=True)
    for v in videos:
        df, meta = track_video(v, args.weights, args.tracker, args.stride,
                               args.imgsz, args.conf, args.max_sec)
        df.to_parquet(args.out / f"{v.stem}_tracks.parquet", index=False)
        (args.out / f"{v.stem}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{v.name}: {meta['n_tracks']} tracks, speed_ratio={meta['speed_ratio']:.2f}, "
              f"short tracks={meta['short_track_share']:.0%}")


if __name__ == "__main__":
    main()
