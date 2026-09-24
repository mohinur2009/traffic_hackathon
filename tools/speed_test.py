"""Compare model size and frame stride on the first N seconds of one video.

Usage:
    python tools/speed_test.py --video samples/a.mp4 --max-sec 60
Prints a markdown table and saves it to reports/speed_table.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from track import track_video  # noqa: E402

CONFIGS = [  # (weights, stride)
    ("yolo11n.pt", 1), ("yolo11s.pt", 1), ("yolo11s.pt", 2),
    ("yolo11s.pt", 3), ("yolo11m.pt", 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--max-sec", type=float, default=60)
    args = ap.parse_args()

    lines = ["| model | stride | device | speed_ratio | tracks | median track (s) | short tracks |",
             "|---|---|---|---|---|---|---|"]
    for weights, stride in CONFIGS:
        _, m = track_video(args.video, weights=weights, stride=stride, max_sec=args.max_sec)
        lines.append(f"| {weights} | {stride} | {m['device']} | {m['speed_ratio']:.2f} | "
                     f"{m['n_tracks']} | {m['median_track_sec']:.1f} | {m['short_track_share']:.0%} |")
        print(lines[-1])

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "speed_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
