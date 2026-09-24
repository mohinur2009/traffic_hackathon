"""Look at cached tracks with your own eyes.

Makes, in reports/:
  <video>_trajectories.jpg  every track's path drawn on a frame (colour = class)
  <video>_counts.png        objects per second, per class
  <video>_clip.mp4          first N seconds with boxes + track IDs (spot ID switches)

Usage:
    python tools/check_tracks.py --video samples/a.mp4 --cache cache/ --clip-sec 20
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"car": (255, 160, 0), "bus": (0, 200, 255), "truck": (0, 120, 255),
          "motorcycle": (255, 0, 200), "bicycle": (0, 255, 120), "person": (0, 0, 255)}


def read_frame(video: Path, t_sec: float):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Cannot read frame at {t_sec}s from {video}")
    return frame


def draw_trajectories(df, frame, out_path):
    img = frame.copy()
    for (_, name), tr in df.sort_values("frame").groupby(["track_id", "cls_name"]):
        pts = tr[["cx", "cy"]].to_numpy().astype(int)
        if len(pts) > 1:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, COLORS.get(name, (200, 200, 200)), 1)
    cv2.imwrite(str(out_path), img)


def plot_counts(df, out_path):
    df = df.assign(sec=df["t_sec"].astype(int))
    per_frame = df.groupby(["sec", "frame", "cls_name"]).size()
    counts = per_frame.groupby(["sec", "cls_name"]).mean().unstack(fill_value=0)
    ax = counts.plot(figsize=(10, 3.5), linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("objects in frame")
    ax.set_title("Objects per second by class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def write_clip(df, video, meta, clip_sec, out_path):
    cap = cv2.VideoCapture(str(video))
    fps, stride = meta["fps"], meta["stride"]
    size = (meta["width"], meta["height"])
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, size)
    by_frame = dict(tuple(df.groupby("frame")))
    for idx in range(int(clip_sec * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            continue
        for r in by_frame.get(idx, pd.DataFrame()).itertuples():
            c = COLORS.get(r.cls_name, (200, 200, 200))
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 2)
            cv2.putText(frame, f"{r.cls_name} #{r.track_id}", (int(r.x1), int(r.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        cv2.putText(frame, f"t={idx / fps:.1f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--clip-sec", type=float, default=20)
    args = ap.parse_args()

    stem = args.video.stem
    df = pd.read_parquet(args.cache / f"{stem}_tracks.parquet")
    meta = json.loads((args.cache / f"{stem}_meta.json").read_text())
    out = Path("reports")
    out.mkdir(exist_ok=True)

    draw_trajectories(df, read_frame(args.video, meta["processed_sec"] / 2), out / f"{stem}_trajectories.jpg")
    plot_counts(df, out / f"{stem}_counts.png")
    write_clip(df, args.video, meta, args.clip_sec, out / f"{stem}_clip.mp4")
    print(f"Saved to {out}/: trajectories, counts, clip. "
          f"{meta['n_tracks']} tracks, short tracks {meta['short_track_share']:.0%}")


if __name__ == "__main__":
    main()
