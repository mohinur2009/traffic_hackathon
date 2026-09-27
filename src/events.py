"""Part A - traffic event detection (simple, rule-based).

Reuses the Part B pipeline (src/risk.py: YOLO11m + ByteTrack, road area,
bird's-eye metres, speeds, time-to-collision) on frames sampled every 0.2 s,
and turns it into two event types:

  near_miss        risk score >= 0.5 (a confirmed close conflict, TTC <~ 1 s)
  stopped_vehicle  a vehicle that was driving stops on the road for >= 10 s
                   and is not waiting in a queue

A time guard stops reading the video after TIME_LIMIT (0.7) x its duration, so
Part A + Part B stay inside the 3x time budget; events found so far are kept.
"""
from __future__ import annotations

import math
import time

import cv2

from src.risk import RiskEstimator

SAMPLE_S = 0.2            # analyse one frame every 0.2 s
TIME_LIMIT = 0.7          # stop after 0.7 x video duration (Part B needs the rest)

# near_miss
NM_THRESHOLD = 0.5        # risk score line
NM_MERGE_GAP = 1.0        # merge alarms closer than 1 s
NM_MIN_LEN = 0.4          # drop blips shorter than 0.4 s
NM_PAD_BEFORE, NM_PAD_AFTER = 0.5, 1.0

# stopped_vehicle
VEHICLES = ("car", "bus", "truck")
STOP_SPEED = 0.5          # m/s: below this = standing
MOVING_SPEED = 2.0        # m/s: must have been seen driving before stopping
MIN_STOP = 10.0           # s (task definition)
MAX_GAP = 1.0             # s: small detection gaps inside a stop are allowed
QUEUE_RADIUS = 12.0       # m
QUEUE_OTHERS = 2          # >= 2 other stopped vehicles nearby = queue
QUEUE_SHARE = 0.5         # queued for at least half of the stop -> not an event


def detect_events(video_path: str) -> list[list]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {"video_id": str(video_path), "fps": fps, "n_frames": n_frames,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    duration = n_frames / fps if n_frames else 0.0

    est = RiskEstimator()
    est.reset(meta)
    est.stride = 1                                  # we already feed only sampled frames
    every = max(1, round(fps * SAMPLE_S))

    risk, recs = [], []                             # (t, score), (t, id, cls, speed, gx, gy)
    t0, idx = time.perf_counter(), 0
    while True:
        if idx % every:
            if not cap.grab():                      # skipped frame: grab is cheaper than read
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        risk.append((t, est.step(frame, t)))
        for o in est.objs:
            recs.append((t, o["id"], o["cls"], math.hypot(*o["v"]),
                         float(o["g"][0]), float(o["g"][1])))
        idx += 1
        if duration and time.perf_counter() - t0 > TIME_LIMIT * duration:
            break                                   # stay inside the time budget
    cap.release()
    end = duration or (idx / fps)
    return events_from_records(risk, recs, end)


def events_from_records(risk, recs, duration) -> list[list]:
    events = [[s, e, "near_miss"] for s, e in near_miss_segments(risk, duration)]
    events += [[s, e, "stopped_vehicle"] for s, e in stopped_segments(recs, duration)]
    return [[round(s, 2), round(e, 2), lab] for s, e, lab in events if e - s > 0.05]


def _merge(segs, gap=0.0):
    out = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def near_miss_segments(risk, duration):
    segs, start, last = [], None, None
    for t, s in risk:
        if s >= NM_THRESHOLD:
            if start is None:
                start = t
            last = t
        elif start is not None:
            segs.append([start, last])
            start = None
    if start is not None:
        segs.append([start, last])
    segs = [[s, e] for s, e in _merge(segs, NM_MERGE_GAP) if e - s >= NM_MIN_LEN]
    padded = [[max(0.0, s - NM_PAD_BEFORE), min(duration, e + NM_PAD_AFTER)] for s, e in segs]
    return _merge(padded)


def stopped_segments(recs, duration):
    by_time, tracks = {}, {}
    for t, tid, cls, spd, gx, gy in recs:
        if cls in VEHICLES:
            by_time.setdefault(t, []).append((tid, spd, gx, gy))
            tracks.setdefault(tid, []).append((t, spd))

    # queued? = at least QUEUE_OTHERS other stopped vehicles within QUEUE_RADIUS
    queued = {}
    for t, vs in by_time.items():
        stopped = [(tid, gx, gy) for tid, spd, gx, gy in vs if spd < STOP_SPEED]
        for tid, gx, gy in stopped:
            near = sum(1 for o, ox, oy in stopped
                       if o != tid and math.hypot(ox - gx, oy - gy) <= QUEUE_RADIUS)
            queued[(tid, t)] = near >= QUEUE_OTHERS

    segs = []
    for tid, pts in tracks.items():
        pts.sort()
        was_moving, run = False, []
        for t, spd in pts + [(math.inf, math.inf)]:          # sentinel closes the last run
            if spd < STOP_SPEED and (not run or t - run[-1] <= MAX_GAP):
                if was_moving:
                    run.append(t)
                continue
            if run and run[-1] - run[0] >= MIN_STOP:
                share = sum(queued.get((tid, x), False) for x in run) / len(run)
                if share < QUEUE_SHARE:
                    segs.append([run[0], min(duration, run[-1])])
            run = []
            if spd < STOP_SPEED and t != math.inf:
                if was_moving:
                    run = [t]
            if spd > MOVING_SPEED:
                was_moving = True
    return _merge(segs)
