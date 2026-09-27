"""Part B - causal accident-risk estimator (P1: perception + risk).

The harness calls ``reset(meta)`` once per video and then ``step(frame, t_sec)``
for every frame, in order. Everything here uses ONLY frames already seen.

Pipeline, about every 0.1 s:
    YOLO11m + ByteTrack  ->  riders / hidden / edge / road-area filters
    ->  feet point to metres (bird's-eye view from a measured crosswalk)
    ->  speed from the last 1 s of positions  ->  time-to-collision (TTC)
        for every pair  ->  risk = sigmoid(TTC), averaged over the last 0.3 s

All rules are in seconds and metres, so 25 fps and 30 fps videos both work,
and the camera settings are rescaled to the real frame size in ``reset``.
"""
from __future__ import annotations

import math
import random
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "yolo11m.pt"


class RiskEstimator:
    """P(an accident starts within the next 5 s), from past frames only."""

    # ---------- camera settings (measured on a 3840x2160 frame) ----------
    # Road area: an object counts only if its feet are inside this polygon.
    ROAD = np.float32([[190, 200], [1700, 420], [2660, 610], [3360, 755], [3840, 990],
                       [3840, 2160], [0, 2160], [0, 1650], [350, 1510], [650, 1290],
                       [0, 1150], [0, 730]])
    # Bird's-eye view: corners of the main crosswalk in the video (TL, TR, BR, BL) and
    # its real size, measured on satellite imagery (26.2 m x 4.5 m).
    IMG_PTS = np.float32([[672, 1262], [3488, 914], [3660, 976], [774, 1356]])
    GND_PTS = np.float32([[0, 0], [26.2, 0], [26.2, 4.5], [0, 4.5]])

    # ---------- detection ----------
    KEEP = [0, 1, 2, 3, 5, 7]          # COCO: person, bicycle, car, motorcycle, bus, truck
    IMGSZ = 1280                        # frames are 4K; 640 loses small far objects
    CONF = 0.3
    STEP_S = 0.1                        # run YOLO about every 0.1 s
    EDGE_PX = 20                        # boxes touching the picture edge are cut -> ignored

    # ---------- motion ----------
    WIN = 0.5                           # speed = mean position (last 0.5 s) vs (0.5 s before)
    MAX_SPEED = 25.0                    # m/s; faster = measurement error / ID jump
    ZONE_Y = -25.0                      # only within ~25 m of the crosswalk (reliable metres)

    # ---------- pair rules ----------
    RADIUS = {"person": 0.5, "rider": 0.7, "bicycle": 0.7, "motorcycle": 0.7,
              "car": 1.2, "truck": 1.5, "bus": 1.6}      # "touching" size in metres
    VEHICLES = ("car", "bus", "truck")
    VEH_MIN_SPEED = 2.0                 # a vehicle must move >= 2 m/s (7 km/h) to be dangerous
    RIDER_MIN_SPEED = 3.0               # a rider alone counts only above 3 m/s
    MIN_CLOSING = 1.0                   # any pair must approach at >= 1 m/s
    SAME_LANE_CLOSING = 3.0             # in line (same lane / heading at it): >= 3 m/s
    LANE_GAP = 1.8                      # both moving the same way: > 1.8 m sideways = next lane
    PASS_GAP = 1.3                      # passing something (almost) standing: > 1.3 m = normal
    START_GAP = 0.5                     # must start > 0.5 m further apart than "touching"
    CONFIRM = 3                         # TTC found 3 times in a row (~0.3 s), not jumping up
    HORIZON = 3.0                       # look 3 s ahead

    # ---------- score ----------
    T0, K = 1.0, 2.0                    # TTC 1 s -> 0.5 (alarm line); 0.5 s -> 0.73; 1.5 s -> 0.27
    SMOOTH_S = 0.3                      # mean of the raw score over the last 0.3 s

    def __init__(self, weights: str | Path = WEIGHTS):
        self.weights = Path(weights)

    # ------------------------------------------------------------------ API
    def reset(self, meta: dict) -> None:
        from ultralytics import YOLO    # imported here so Part A can import this file cheaply
        import torch

        if not self.weights.exists():
            raise FileNotFoundError(f"{self.weights} not found - run weights/download.sh once")
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        self.fps = float(meta.get("fps") or 25.0)
        self.w, self.h = int(meta["width"]), int(meta["height"])
        s = np.float32([self.w / 3840, self.h / 2160])
        self.road = (self.ROAD * s).astype(np.int32).reshape(-1, 1, 2)
        self.HM = cv2.getPerspectiveTransform(self.IMG_PTS * s, self.GND_PTS)
        self.stride = max(1, round(self.fps * self.STEP_S))
        self.model = YOLO(str(self.weights))            # fresh model = fresh tracker per video
        self.n = 0
        self.hist, self.labels, self.last_seen, self.vhist = {}, {}, {}, {}
        self.pairs = {}                                 # (id_a, id_b) -> (t, ttc, count)
        self.raw_hist = deque()
        self.score = 0.0
        self.why = None                                 # reason for the latest score (for demos)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.n += 1
        if (self.n - 1) % self.stride != 0:
            return self.score                           # skipped frame: repeat last score
        res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.IMGSZ,
                               classes=self.KEEP, conf=self.CONF, verbose=False)[0]
        objs = self._objects(res, t_sec)
        raw = self._raw_risk(objs, t_sec)
        self.raw_hist.append((t_sec, raw))
        while self.raw_hist and self.raw_hist[0][0] < t_sec - self.SMOOTH_S:
            self.raw_hist.popleft()
        self.score = float(np.clip(np.mean([r for _, r in self.raw_hist]), 0.0, 1.0))
        return self.score

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _overlap(a, b) -> float:
        """Share of box a covered by box b."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        return ix * iy / area

    def _forget(self, t: float) -> None:
        for tid in [k for k, ts in self.last_seen.items() if ts < t - 2.0]:
            for d in (self.hist, self.labels, self.last_seen, self.vhist):
                d.pop(tid, None)
        for key in [k for k, v in self.pairs.items() if v[0] < t - 1.0]:
            self.pairs.pop(key, None)

    def _objects(self, res, t: float) -> list[dict]:
        """Tracked road users on the road, in metres, with a reliable speed."""
        b = res.boxes
        if b is None or b.id is None or len(b) == 0:
            self._forget(t)
            return []
        xyxy = b.xyxy.cpu().numpy()
        ids = b.id.int().cpu().numpy()
        cls = [self.model.names[int(c)] for c in b.cls.cpu().numpy()]

        # rider = a person sitting on a bike box (the bike box itself is then dropped)
        bikes = [k for k, c in enumerate(cls) if c in ("bicycle", "motorcycle")]
        drop = set()
        for k, c in enumerate(cls):
            if c == "person":
                for m in bikes:
                    if self._overlap(xyxy[k], xyxy[m]) >= 0.3:
                        cls[k] = "rider"
                        drop.add(m)

        e = self.EDGE_PX
        objs = []
        for k in range(len(ids)):
            if k in drop:
                continue
            x1, y1, x2, y2 = xyxy[k]
            if x1 <= e or y1 <= e or x2 >= self.w - e or y2 >= self.h - e:
                continue                                # cut by the picture edge
            fx, fy = (x1 + x2) / 2, y2                  # feet = where it touches the road
            if self._hidden(k, fx, fy, y2, xyxy, drop):
                continue                                # feet behind something in front
            if cv2.pointPolygonTest(self.road, (float(fx), float(fy)), False) < 0:
                continue                                # not on the road
            gx, gy = cv2.perspectiveTransform(np.float32([[[fx, fy]]]), self.HM)[0, 0]
            if gy <= self.ZONE_Y:
                continue                                # too far to measure well
            tid = int(ids[k])
            h = self.hist.setdefault(tid, deque())
            if h:
                tp, xp, yp = h[-1]
                if t > tp and math.hypot(gx - xp, gy - yp) / (t - tp) > self.MAX_SPEED:
                    h.clear()                           # ID jumped to another object
                    self.labels[tid] = deque(maxlen=11)
                    self.vhist[tid] = deque(maxlen=5)
            h.append((t, float(gx), float(gy)))
            while h and h[0][0] < t - 2 * self.WIN:
                h.popleft()
            lab = self.labels.setdefault(tid, deque(maxlen=11))   # type = vote over ~1 s
            lab.append(cls[k])
            self.last_seen[tid] = t
            pos, vel = self._motion(h)
            if vel is None:
                continue                                # too new to know its speed
            vh = self.vhist.setdefault(tid, deque(maxlen=5))
            vh.append(vel)
            vel = np.median(np.array(vh), axis=0)       # steadier speed
            objs.append({"id": tid, "cls": Counter(lab).most_common(1)[0][0],
                         "g": pos, "v": vel,
                         "box": [float(x1), float(y1), float(x2), float(y2)]})
        self._forget(t)
        return objs

    @staticmethod
    def _hidden(k, fx, fy, y2, xyxy, drop) -> bool:
        for m in range(len(xyxy)):
            if m == k or m in drop:
                continue
            X1, Y1, X2, Y2 = xyxy[m]
            if Y2 > y2 and X1 <= fx <= X2 and Y1 <= fy <= Y2:
                return True
        return False

    def _motion(self, h):
        """Mean position of the last 0.5 s vs the 0.5 s before -> (position now, velocity)."""
        t_now = h[-1][0]
        new = [p for p in h if p[0] > t_now - self.WIN]
        old = [p for p in h if t_now - 2 * self.WIN < p[0] <= t_now - self.WIN]
        if len(new) < 2 or len(old) < 2:
            return None, None
        a, b = np.mean(new, axis=0), np.mean(old, axis=0)
        dt = a[0] - b[0]
        if dt <= 0:
            return None, None
        v = (a[1:] - b[1:]) / dt
        if math.hypot(v[0], v[1]) > self.MAX_SPEED:
            return None, None
        return a[1:] + v * (t_now - a[0]), v

    def _raw_risk(self, objs: list[dict], t: float) -> float:
        """Smallest confirmed TTC over all pairs -> score in [0, 1]."""
        best, info = None, None
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                ttc, extra = self._pair_ttc(objs[i], objs[j], t)
                if ttc is not None and (best is None or ttc < best):
                    best, info = ttc, extra
        self.why = info
        if best is None:
            return 0.0
        return 1.0 / (1.0 + math.exp(self.K * (best - self.T0)))

    def _pair_ttc(self, A: dict, B: dict, t: float):
        sa, sb = math.hypot(*A["v"]), math.hypot(*B["v"])
        if sa <= 0.5 and sb <= 0.5:
            return None, None                           # both standing still
        ca, cb = A["cls"], B["cls"]
        if ca == "person" and cb == "person":
            return None, None                           # two pedestrians
        has_vehicle = ((ca in self.VEHICLES and sa > self.VEH_MIN_SPEED) or
                       (cb in self.VEHICLES and sb > self.VEH_MIN_SPEED))
        fast_rider = ((ca == "rider" and sa > self.RIDER_MIN_SPEED) or
                      (cb == "rider" and sb > self.RIDER_MIN_SPEED))
        if not (has_vehicle or fast_rider):
            return None, None

        p = B["g"] - A["g"]                             # where B is, seen from A
        v = B["v"] - A["v"]                             # how B moves, seen from A
        R = self.RADIUS.get(ca, 1.0) + self.RADIUS.get(cb, 1.0)
        dist = math.hypot(p[0], p[1])
        closing = -float(p @ v) / max(dist, 1e-6)       # m/s they get closer
        if closing < self.MIN_CLOSING or dist < R + self.START_GAP:
            return None, None

        # lane rules: side-by-side traffic and normal following are not danger
        d, gap = None, self.LANE_GAP
        if sa > 1.0 and sb > 1.0 and float(A["v"] @ B["v"]) / (sa * sb) > 0.87:
            d = (A["v"] + B["v"]) / np.linalg.norm(A["v"] + B["v"])   # both going the same way
        elif max(sa, sb) > 1.0 and min(sa, sb) < 1.0:
            fast = A["v"] if sa > sb else B["v"]
            d = fast / np.linalg.norm(fast)                            # passing a (near) standing one
            gap = self.PASS_GAP
        if d is not None:
            side = p[0] * d[1] - p[1] * d[0]
            side_closing = -np.sign(side) * (v[0] * d[1] - v[1] * d[0])
            if abs(side) > gap and side_closing < 1.0:
                return None, None                       # next to each other, no sideways move
            if abs(side) <= gap and closing < self.SAME_LANE_CLOSING:
                return None, None                       # in line, approaching slowly

        c = float(p @ p) - R * R
        a = float(v @ v)
        bb = 2 * float(p @ v)
        disc = bb * bb - 4 * a * c
        if disc < 0:
            return None, None                           # will pass without touching
        ttc = (-bb - math.sqrt(disc)) / (2 * a)
        if ttc > self.HORIZON:
            return None, None

        # the danger must be consistent: found several times in a row, TTC not jumping up
        key = (min(A["id"], B["id"]), max(A["id"], B["id"]))
        prev = self.pairs.get(key)
        count = prev[2] + 1 if prev and t - prev[0] <= 0.35 and ttc <= prev[1] + 0.3 else 1
        self.pairs[key] = (t, ttc, count)
        if count < self.CONFIRM:
            return None, None
        return ttc, {"ttc": round(ttc, 2), "a_id": A["id"], "a_cls": ca,
                     "b_id": B["id"], "b_cls": cb, "dist_m": round(dist, 1),
                     "closing_ms": round(closing, 1), "speed_a": round(sa, 1),
                     "speed_b": round(sb, 1), "box_a": A["box"], "box_b": B["box"]}
