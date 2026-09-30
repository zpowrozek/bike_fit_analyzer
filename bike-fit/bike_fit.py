#!/usr/bin/env python3
"""Grade a road-bike fit from a side-on video of the rider pedaling on a trainer.

Pipeline
    1. ffmpeg normalizes the phone video (bakes in rotation, H.264, caps fps/size).
    2. YOLO11x-pose (ultralytics) finds 17 COCO keypoints per frame; we keep the
       highest-confidence person and the camera-side shoulder/elbow/wrist/hip/knee/ankle.
    3. Pedaling is detected (knee oscillating, hips planted on the saddle), and
       bottom-dead-center (BDC = knee most extended) is found on every stroke.
    4. Angles at BDC are median-ed across strokes and graded green/amber/red.
    5. A colored skeleton is drawn over the clip (supervision), the BDC still and a
       plain-English report are saved, and the overlay is re-encoded to H.264.

Grading ranges (dynamic, road, measured while pedaling)
    Knee flexion @BDC   30-40 deg  Holmes static 25-35 deg, +~8 deg when measured
                                   dynamically on video (Bini et al. 2014: static knee
                                   8.2 +/- 5 deg smaller at 6 o'clock). Field averages
                                   knee 36 +/- 7 deg (PMC9219349).
    Elbow flexion       15-30 deg  field average 19 +/- 8 deg (PMC9219349).
    Torso from horiz.   40-50 deg  practitioner range (bikefitadviser.com, part 3).
    Shoulder angle      80-95 deg  torso-to-upper-arm; practitioner range.
    Hip angle @BDC      85-110 deg torso-to-thigh, report only (flexibility dependent).
                                   This range matches the hip at the BOTTOM of the stroke;
                                   at the top a road hip is ~45-60 deg, so the top-of-stroke
                                   angle is reported as a number, not graded.
A +/-2.5 deg tolerance band around each green zone is graded amber, because a phone
video only resolves joint angles to a few degrees.

Usage
    python bike_fit.py my_video.mov --height-cm 180
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from scipy.signal import find_peaks, medfilt

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "yolo11x-pose.pt"

# COCO-17 keypoint indices (ultralytics pose order).
SIDES = {
    "left": dict(shoulder=5, elbow=7, wrist=9, hip=11, knee=13, ankle=15),
    "right": dict(shoulder=6, elbow=8, wrist=10, hip=12, knee=14, ankle=16),
}
JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")

KP_CONF = 0.5  # ignore keypoints the model is less sure of than this
TOL = 2.5  # deg of amber band either side of a green zone

# name: (low, high, label, graded?)
RANGES = {
    "knee": (30.0, 40.0, "Knee bend at bottom", True),
    "torso": (40.0, 50.0, "Torso from horizontal", True),
    "elbow": (15.0, 30.0, "Elbow bend", True),
    "shoulder": (80.0, 95.0, "Shoulder angle", True),
    "hip": (85.0, 110.0, "Hip angle at bottom", False),
}

# Pedaling detection.
WINDOW_S = 2.0  # sliding window used to decide "is this pedaling?"
MIN_KNEE_RANGE = 25.0  # deg of knee swing inside the window
MAX_HIP_SPREAD = 0.12  # hip wander (std, in thigh lengths) allowed while seated
MIN_SEGMENT_S = 2.0  # shortest pedaling stretch we trust
MAX_RPM = 150.0
STROKE_PROMINENCE = 15.0  # deg a BDC dip must stand out from the stroke

# Anthropometrics (Winter): thigh 0.245 x height, shank 0.246 x height.
LEG_FRACTION = 0.491
DEFAULT_HEIGHT_CM = 175.0

GREEN = sv.Color(r=40, g=200, b=70)
AMBER = sv.Color(r=255, g=150, b=0)
RED = sv.Color(r=235, g=40, b=40)
GRAY = sv.Color(r=150, g=150, b=150)
WHITE = sv.Color(r=255, g=255, b=255)
BLACK = sv.Color(r=0, g=0, b=0)
GRADE_COLOR = {"green": GREEN, "amber": AMBER, "red": RED, "n/a": GRAY}
GRADE_RANK = {"n/a": -1, "green": 0, "amber": 1, "red": 2}


# --------------------------------------------------------------------------- video io


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def normalize_video(src: Path, dst: Path, max_fps: float = 60.0, max_side: int = 1920) -> None:
    """Re-encode to H.264 with rotation baked in, at most max_fps and max_side px."""
    rate = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=avg_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(src)])
    num, _, den = (rate.split() or ["30/1"])[0].partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    filters = [f"scale=w='min({max_side},iw)':h='min({max_side},ih)'"
               ":force_original_aspect_ratio=decrease:force_divisible_by=2"]
    if fps > max_fps + 0.5:
        filters.append(f"fps={max_fps:g}")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", ",".join(filters), "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)])


def to_h264(src: Path, dst: Path, fps: float) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-r", f"{fps:.6f}", "-i", str(src), "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(dst)])


def run_pose(video: Path, model_path: str, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO11-pose on every frame -> (F,17,2) pixel xy and (F,17) confidence."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    total = sv.VideoInfo.from_video_path(str(video)).total_frames
    xys, confs = [], []
    t0 = time.time()
    for i, frame in enumerate(sv.get_video_frames_generator(str(video))):
        r = model.predict(frame, imgsz=imgsz, conf=0.25, verbose=False)[0]
        xy = np.full((17, 2), np.nan, np.float32)
        cf = np.zeros(17, np.float32)
        if r.keypoints is not None and r.keypoints.conf is not None and len(r.boxes):
            best = int(r.boxes.conf.argmax())  # highest-confidence person
            xy = r.keypoints.xy[best].cpu().numpy()
            cf = r.keypoints.conf[best].cpu().numpy()
        xys.append(xy)
        confs.append(cf)
        if (i + 1) % 25 == 0 or i + 1 == total:
            rate = (i + 1) / (time.time() - t0)
            eta = (total - i - 1) / rate if total else 0
            print(f"  pose {i + 1}/{total} frames  ({rate:.1f} fps, ~{eta:.0f}s left)", flush=True)
    return np.stack(xys), np.stack(confs)


# --------------------------------------------------------------------------- geometry


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b (deg) between b->a and b->c, per frame."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def torso_from_horizontal(shoulder: np.ndarray, hip: np.ndarray) -> np.ndarray:
    rise = hip[:, 1] - shoulder[:, 1]  # image y points down
    run_ = np.abs(shoulder[:, 0] - hip[:, 0])
    return np.degrees(np.arctan2(rise, run_))


def reject_length_outliers(series: np.ndarray, a: np.ndarray, b: np.ndarray,
                           c: np.ndarray, tol: float) -> np.ndarray:
    """NaN frames where either segment of a-b-c is far from its clip-median length."""
    l1 = np.linalg.norm(a - b, axis=-1)
    l2 = np.linalg.norm(b - c, axis=-1)
    bad = (np.abs(l1 / np.nanmedian(l1) - 1) > tol) | (np.abs(l2 / np.nanmedian(l2) - 1) > tol)
    out = series.copy()
    out[bad] = np.nan
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True stretches."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_gaps(x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return x.copy()
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def near(series: np.ndarray, i: int, r: int = 1) -> float:
    """Median of series around frame i (tolerates one blurry frame)."""
    w = series[max(0, i - r): i + r + 1]
    return float(np.nanmedian(w)) if np.any(~np.isnan(w)) else float("nan")


# --------------------------------------------------------------------------- analysis


def pedaling_mask(knee: np.ndarray, hip: np.ndarray, thigh_px: float, fps: float) -> np.ndarray:
    """True where the rider is seated and actively pedaling.

    Pedaling = the knee swings through a big range while the hip stays planted on
    the saddle. Mounting, walking around and standing next to the bike move the hip;
    coasting and standing still don't swing the knee.
    """
    n = len(knee)
    half = max(3, int(round(fps * WINDOW_S / 2)))
    mask = np.zeros(n, bool)
    for t in range(n):
        a, b = max(0, t - half), min(n, t + half + 1)
        k = knee[a:b][~np.isnan(knee[a:b])]
        h = hip[a:b][~np.isnan(hip[a:b, 0])]
        if len(k) < 0.5 * (b - a) or len(h) < 0.5 * (b - a):
            continue
        swing = np.percentile(k, 95) - np.percentile(k, 5)
        wander = h.std(axis=0).max() / thigh_px
        mask[t] = swing >= MIN_KNEE_RANGE and wander <= MAX_HIP_SPREAD
    # Trim the window's reach at each end, then drop stretches too short to trust.
    trim = int(round(fps * 0.5))
    out = np.zeros(n, bool)
    for s, e in runs(mask):
        s, e = s + trim, e - trim
        if e - s >= fps * MIN_SEGMENT_S:
            out[s:e] = True
    return out


def find_strokes(knee: np.ndarray, pedaling: np.ndarray, fps: float) -> tuple[list[int], list[int]]:
    """Frames of BDC (knee most extended) and TDC (most flexed), pedaling only."""
    bdc, tdc = [], []
    distance = max(1, int(fps * 60.0 / MAX_RPM))
    for s, e in runs(pedaling):
        seg = medfilt(fill_gaps(knee[s:e]), 3)
        if np.isnan(seg).any():
            continue
        mid = np.median(seg)
        lows, _ = find_peaks(-seg, distance=distance, prominence=STROKE_PROMINENCE)
        highs, _ = find_peaks(seg, distance=distance, prominence=STROKE_PROMINENCE)
        for p in lows:
            if seg[p] >= mid:
                continue
            # Smoothing can shift the dip a frame; snap to the straightest real frame.
            lo, hi = max(s, s + p - 1), min(e, s + p + 2)
            raw = knee[lo:hi]
            if np.any(~np.isnan(raw)):
                bdc.append(lo + int(np.nanargmin(raw)))
        tdc += [s + int(p) for p in highs if seg[p] > mid]
    return bdc, tdc


def grade(value: float, lo: float, hi: float) -> str:
    if np.isnan(value):
        return "n/a"
    if lo <= value <= hi:
        return "green"
    if lo - TOL <= value <= hi + TOL:
        return "amber"
    return "red"


def worse(*grades: str) -> str:
    return max(grades, key=lambda g: GRADE_RANK[g])


def saddle_change_mm(flex_now: float, flex_target: float, thigh_mm: float, shank_mm: float) -> float:
    """Change in hip-to-ankle distance (~saddle height) to move knee flexion. +ve = raise."""
    def reach(flex):
        th = np.radians(180.0 - flex)
        return np.sqrt(thigh_mm**2 + shank_mm**2 - 2 * thigh_mm * shank_mm * np.cos(th))
    return float(reach(flex_target) - reach(flex_now))


def round5(mm: float) -> int:
    return int(max(5, 5 * round(abs(mm) / 5)))


def analyze(xy: np.ndarray, conf: np.ndarray, fps: float, height_cm: float | None = None) -> dict:
    # Camera-side limbs are the ones the model sees most confidently.
    score = {s: float(np.mean(conf[:, list(ix.values())])) for s, ix in SIDES.items()}
    side = max(score, key=score.get)
    J = {}
    for name, k in SIDES[side].items():
        p = xy[:, k].astype(np.float64).copy()
        p[conf[:, k] < KP_CONF] = np.nan
        J[name] = p

    knee = 180.0 - angle_at(J["hip"], J["knee"], J["ankle"])
    knee = reject_length_outliers(knee, J["hip"], J["knee"], J["ankle"], 0.25)
    elbow = 180.0 - angle_at(J["shoulder"], J["elbow"], J["wrist"])
    elbow = reject_length_outliers(elbow, J["shoulder"], J["elbow"], J["wrist"], 0.30)
    shoulder = angle_at(J["hip"], J["shoulder"], J["elbow"])
    torso = torso_from_horizontal(J["shoulder"], J["hip"])
    hip = angle_at(J["shoulder"], J["hip"], J["knee"])
    series = dict(knee=knee, elbow=elbow, shoulder=shoulder, torso=torso, hip=hip)

    thigh_px = float(np.nanmedian(np.linalg.norm(J["hip"] - J["knee"], axis=-1)))
    shank_px = float(np.nanmedian(np.linalg.norm(J["knee"] - J["ankle"], axis=-1)))
    if np.isnan(thigh_px):
        raise SystemExit("Couldn't see a hip and knee clearly in this video. "
                         "Film side-on with your whole body in frame.")

    pedaling = pedaling_mask(knee, J["hip"], thigh_px, fps)
    bdc, tdc = find_strokes(knee, pedaling, fps)

    # Knee: the single BDC frame (it's the extreme, so averaging neighbours would read
    # too bent). Upper body barely moves, so +/-1 frame smooths out blur.
    per_stroke = {m: [near(series[m], i) for i in bdc] for m in ("torso", "elbow", "shoulder", "hip")}
    per_stroke["knee"] = [float(knee[i]) for i in bdc]
    hip_top = [near(hip, i) for i in tdc]
    hip_top = float(np.nanmedian(hip_top)) if np.any(~np.isnan(hip_top)) else float("nan")

    metrics = {}
    for m, (lo, hi, label, graded) in RANGES.items():
        vals = np.array(per_stroke[m], float)
        vals = vals[~np.isnan(vals)]
        med = float(np.median(vals)) if len(vals) else float("nan")
        spread = float(np.subtract(*np.percentile(vals, [75, 25]))) if len(vals) > 1 else float("nan")
        metrics[m] = dict(label=label, value=med, n=int(len(vals)), iqr=spread, low=lo, high=hi,
                          graded=graded, grade=grade(med, lo, hi))

    beats = [b - a for a, b in zip(bdc, bdc[1:]) if pedaling[a:b].all()]
    cadence = 60.0 * fps / float(np.median(beats)) if beats else float("nan")

    height_mm = 10.0 * (height_cm or DEFAULT_HEIGHT_CM)
    leg_mm = LEG_FRACTION * height_mm
    thigh_mm = leg_mm * thigh_px / (thigh_px + shank_px)
    shank_mm = leg_mm - thigh_mm

    g = {m: metrics[m]["grade"] for m in metrics}
    colors = {
        "leg": g["knee"],
        "torso": g["torso"],
        "arm": worse(g["shoulder"], g["elbow"]),  # so a closed cockpit shows on the arm
    }
    return dict(
        side=side, fps=fps, n_frames=len(xy), joints=J, series=series, pedaling=pedaling,
        bdc=bdc, tdc=tdc, metrics=metrics, cadence=cadence, colors=colors, hip_top=hip_top,
        height_cm=height_cm, thigh_mm=thigh_mm, shank_mm=shank_mm,
        pedaling_s=float(pedaling.sum() / fps),
        advice=advise(metrics, thigh_mm, shank_mm, height_cm is None, hip_top),
    )


# --------------------------------------------------------------------------- advice


def advise(m: dict, thigh_mm: float, shank_mm: float, height_guessed: bool,
           hip_top: float = float("nan")) -> list[dict]:
    """Ordered fix list: saddle height first (it changes everything else), then reach,
    then bar height."""
    fixes = []

    k = m["knee"]
    v = k["value"]
    if k["grade"] == "n/a":
        fixes.append(dict(area="Saddle height", grade="n/a",
                          text="Couldn't find enough clean pedal strokes to measure the knee."))
    elif k["grade"] == "green":
        fixes.append(dict(area="Saddle height", grade="green",
                          text=f"Knee {v:.0f} deg at the bottom -> saddle height is dialed. Leave it."))
    else:
        low = v > k["high"]
        target = k["high"] - 1.5 if low else k["low"] + 1.5
        mm = round5(saddle_change_mm(v, target, thigh_mm, shank_mm))
        how = "raise" if low else "lower"
        verdict = "too low" if low else "too high"
        if k["grade"] == "amber":
            verdict = "a touch " + verdict
            text = (f"Knee {v:.0f} deg at the bottom -> saddle {verdict}. Optional: {how} ~{mm} mm, "
                    "or leave it if you're comfortable.")
        else:
            text = (f"Knee {v:.0f} deg at the bottom -> saddle {verdict}, {how} ~{mm} mm "
                    f"(aiming for ~{target:.0f} deg). Do it in 5 mm steps and re-film.")
        if height_guessed:
            text += f" (mm estimate assumes a {DEFAULT_HEIGHT_CM:.0f} cm rider.)"
        fixes.append(dict(area="Saddle height", grade=k["grade"], text=text))

    sh, el = m["shoulder"], m["elbow"]
    short = (sh["grade"] != "n/a" and sh["value"] < sh["low"]) or (el["grade"] != "n/a" and el["value"] > el["high"])
    long_ = (sh["grade"] != "n/a" and sh["value"] > sh["high"]) or (el["grade"] != "n/a" and el["value"] < el["low"])
    reach_grade = worse(sh["grade"], el["grade"])
    bits = []
    if sh["grade"] != "n/a":
        bits.append(f"shoulder {sh['value']:.0f} deg")
    if el["grade"] != "n/a":
        bits.append(f"elbow {el['value']:.0f} deg")
    got = ", ".join(bits) or "arms not visible"
    if reach_grade == "n/a":
        text = "Couldn't see your arm clearly enough to judge reach."
    elif reach_grade == "green":
        text = f"{got.capitalize()} -> reach is dialed."
    elif short and not long_:
        text = (f"{got.capitalize()} -> cockpit is closed/scrunched: reach is short or you're "
                "carrying weight on your hands. Try a 10-20 mm longer stem"
                + (" once the saddle is set." if m["knee"]["grade"] in ("amber", "red") else "."))
    elif long_ and not short:
        text = (f"{got.capitalize()} -> you're stretched out (arms near-locked). Try a 10-20 mm "
                "shorter stem, and keep a soft bend in the elbows.")
    else:
        text = (f"{got.capitalize()} -> mixed signals (shoulder and elbow disagree). Usually this "
                "is posture on the hoods, not the bike; relax the shoulders and re-film before changing parts.")
    fixes.append(dict(area="Reach", grade=reach_grade, text=text))

    t = m["torso"]
    if t["grade"] == "n/a":
        text = "Couldn't see your shoulder and hip clearly enough to measure torso angle."
    elif t["grade"] == "green":
        text = f"Torso {t['value']:.0f} deg from horizontal -> bar height is dialed."
    elif t["value"] < t["low"]:
        text = (f"Torso {t['value']:.0f} deg -> low, aggressive front end. Fine if it's comfortable; "
                "if your neck, back or hands complain, raise the bars 10-20 mm (move a spacer under the stem).")
    else:
        text = (f"Torso {t['value']:.0f} deg -> upright. Fine for comfort; for more aero, drop the bars "
                "~10 mm at a time once saddle and reach are set.")
    fixes.append(dict(area="Bar height", grade=t["grade"], text=text))

    h = m["hip"]
    if h["grade"] != "n/a":
        note = "within the usual range" if h["grade"] == "green" else (
            "closed - tight hamstrings/lower back may complain; a higher front end opens it"
            if h["value"] < h["low"] else "open - plenty of room")
        top = f" At the top of the stroke it closes to {hip_top:.0f} deg." if not np.isnan(hip_top) else ""
        fixes.append(dict(area="Hip angle (info only)", grade=h["grade"],
                          text=f"Hip {h['value']:.0f} deg at the bottom -> {note}.{top} "
                               "Flexibility dependent, so not graded."))
    return fixes


# --------------------------------------------------------------------------- drawing


SEGMENTS = [("shoulder", "elbow", "arm"), ("elbow", "wrist", "arm"), ("shoulder", "hip", "torso"),
            ("hip", "knee", "leg"), ("knee", "ankle", "leg")]
# (joint to label, metric shown). Numbers are colored by their own grade; the arm
# line is colored by the worse of shoulder/elbow.
LABELS = [("knee", "knee"), ("elbow", "elbow"), ("shoulder", "shoulder"), ("hip", "torso")]


def pt(p: np.ndarray) -> sv.Point:
    return sv.Point(int(round(p[0])), int(round(p[1])))


def draw_frame(frame: np.ndarray, i: int, res: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w) / 1080.0
    thick = max(2, int(round(6 * s)))
    J = res["joints"]
    pedaling = bool(res["pedaling"][i])
    is_bdc = i in res["_bdc_set"]
    color_of = {k: (GRADE_COLOR[g] if pedaling else GRAY) for k, g in res["colors"].items()}

    for a, b, part in SEGMENTS:
        pa, pb = J[a][i], J[b][i]
        if not (np.isnan(pa).any() or np.isnan(pb).any()):
            sv.draw_line(frame, pt(pa), pt(pb), color=color_of[part], thickness=thick)
    for name in JOINTS:
        p = J[name][i]
        if not np.isnan(p).any():
            c = (int(p[0]), int(p[1]))
            cv2.circle(frame, c, thick + 3, WHITE.as_bgr(), -1, cv2.LINE_AA)
            cv2.circle(frame, c, thick + 3, BLACK.as_bgr(), max(1, thick // 3), cv2.LINE_AA)

    if pedaling:
        for joint, metric in LABELS:
            v, p = res["series"][metric][i], J[joint][i]
            if np.isnan(v) or np.isnan(p).any():
                continue
            anchor = sv.Point(int(p[0] + 60 * s), int(p[1] - 40 * s))
            sv.draw_text(frame, f"{v:.0f}", anchor, text_color=WHITE, text_scale=0.9 * s,
                         text_thickness=max(1, int(2 * s)), text_padding=int(6 * s) + 2,
                         background_color=GRADE_COLOR[res["metrics"][metric]["grade"]])

    draw_hud(frame, res, s)
    banner = "BDC" if is_bdc else (None if pedaling else "not pedaling - ignored")
    if banner:
        sv.draw_text(frame, banner, sv.Point(w // 2, h - int(50 * s)), text_color=WHITE,
                     text_scale=1.2 * s, text_thickness=max(1, int(3 * s)), text_padding=int(10 * s) + 2,
                     background_color=GREEN if is_bdc else GRAY)
    return frame


def draw_hud(frame: np.ndarray, res: dict, s: float) -> None:
    lines = []
    for m, r in res["metrics"].items():
        val = "--" if np.isnan(r["value"]) else f"{r['value']:.0f}"
        tag = "" if r["graded"] else " (info)"
        lines.append((f"{r['label']}: {val} deg  [{r['low']:.0f}-{r['high']:.0f}]{tag}",
                      GRADE_COLOR[r["grade"]] if r["graded"] else WHITE))
    head = f"BIKE FIT  {len(res['bdc'])} strokes"
    if not np.isnan(res["cadence"]):
        head += f"  ~{res['cadence']:.0f} rpm"
    lines.insert(0, (head, WHITE))
    fs, lh, pad = 0.75 * s, int(36 * s) + 4, int(14 * s) + 4
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, int(2 * s)))[0][0] for t, _ in lines)
    rect = sv.Rect(x=pad, y=pad, width=tw + 2 * pad, height=lh * len(lines) + pad)
    sv.draw_filled_rectangle(frame, rect, color=BLACK, opacity=0.55)
    for n, (t, c) in enumerate(lines):
        y = 2 * pad + n * lh + int(10 * s)
        cv2.putText(frame, t, (2 * pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs, c.as_bgr(),
                    max(1, int(2 * s)), cv2.LINE_AA)


def render(video: Path, res: dict, out_dir: Path) -> tuple[Path, Path | None]:
    res["_bdc_set"] = set(res["bdc"])
    info = sv.VideoInfo.from_video_path(str(video))
    # Still: the stroke whose knee angle is closest to the median.
    still_i = None
    if res["bdc"]:
        med = res["metrics"]["knee"]["value"]
        still_i = min(res["bdc"], key=lambda i: abs(near(res["series"]["knee"], i) - med))
    raw, final, still = out_dir / "overlay_raw.mp4", out_dir / "overlay.mp4", out_dir / "bdc_still.jpg"
    with sv.VideoSink(str(raw), video_info=info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(str(video))):
            if i >= res["n_frames"]:
                break
            frame = draw_frame(frame, i, res)
            sink.write_frame(frame)
            if i == still_i:
                cv2.imwrite(str(still), frame)
    to_h264(raw, final, res["fps"])
    raw.unlink()
    return final, (still if still_i is not None else None)


# --------------------------------------------------------------------------- report


def report(res: dict) -> str:
    icon = {"green": "GREEN", "amber": "AMBER", "red": "RED", "n/a": "n/a"}
    L = ["# Bike fit report", ""]
    L.append(f"- Side analysed: **{res['side']}** (camera side)")
    L.append(f"- Pedaling found: **{res['pedaling_s']:.1f} s**, **{len(res['bdc'])} strokes** measured"
             + (f", cadence ~{res['cadence']:.0f} rpm" if not np.isnan(res["cadence"]) else ""))
    if len(res["bdc"]) < 3:
        L.append("- **Warning:** fewer than 3 clean strokes - film at least 10 s of steady seated pedaling.")
    L += ["", "| Measure | You (median) | Green zone | Grade | Stroke spread (IQR) |", "|---|---|---|---|---|"]
    for r in res["metrics"].values():
        val = "--" if np.isnan(r["value"]) else f"{r['value']:.0f} deg"
        iqr = "--" if np.isnan(r["iqr"]) else f"{r['iqr']:.1f} deg"
        gr = icon[r["grade"]] + ("" if r["graded"] else " (info only)")
        L.append(f"| {r['label']} | {val} | {r['low']:.0f}-{r['high']:.0f} deg | {gr} | {iqr} |")
    L += ["", "## What to do (in this order)", ""]
    for n, f in enumerate(res["advice"], 1):
        L.append(f"{n}. **{f['area']}** [{icon[f['grade']]}]: {f['text']}")
    L += ["", "## Notes", "",
          "- Angles are measured at bottom-dead-center (knee most extended) on every stroke and "
          "median-ed, so one blurry frame can't skew them. Only seated pedaling counts.",
          f"- Amber = within {TOL} deg of the green zone; a phone video resolves angles to a few degrees.",
          "- Saddle fore-aft (setback) isn't measured here. Change one thing at a time, then re-film.",
          "- Knee range is dynamic (30-40 deg): Holmes' static 25-35 deg plus the ~8 deg that "
          "video-while-pedaling adds (Bini et al. 2014)."]
    return "\n".join(L) + "\n"


def to_json(res: dict) -> dict:
    return dict(side=res["side"], fps=res["fps"], strokes=len(res["bdc"]), cadence_rpm=res["cadence"],
                pedaling_seconds=res["pedaling_s"], bdc_frames=[int(i) for i in res["bdc"]],
                tdc_frames=[int(i) for i in res["tdc"]],
                metrics=res["metrics"], hip_angle_top=res["hip_top"], advice=res["advice"],
                height_cm=res["height_cm"])


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, help="output folder (default results/<video name>)")
    ap.add_argument("--height-cm", type=float, help="your height, for the saddle mm estimate")
    ap.add_argument("--model", default=str(HERE / DEFAULT_MODEL))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fresh", action="store_true", help="re-run pose even if cached")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found - install ffmpeg first.")
    if not args.video.exists():
        sys.exit(f"No such file: {args.video}")
    out = args.out or HERE / "results" / args.video.stem
    out.mkdir(parents=True, exist_ok=True)

    print("1/4 Preparing video...")
    norm = out / "input.mp4"
    normalize_video(args.video, norm)
    fps = sv.VideoInfo.from_video_path(str(norm)).fps

    cache = out / "keypoints.npz"
    key = f"{args.video.resolve()}|{args.video.stat().st_mtime}|{Path(args.model).name}|{args.imgsz}"
    if cache.exists() and not args.fresh and str(np.load(cache)["key"]) == key:
        print("2/4 Pose: using cached keypoints")
        d = np.load(cache)
        xy, conf = d["xy"], d["conf"]
    else:
        print(f"2/4 Finding your joints with {Path(args.model).name} (slow on CPU)...")
        xy, conf = run_pose(norm, args.model, args.imgsz)
        np.savez_compressed(cache, xy=xy, conf=conf, key=key)

    print("3/4 Measuring angles at the bottom of each stroke...")
    res = analyze(xy, conf, fps, args.height_cm)

    print("4/4 Drawing the overlay...")
    video, still = render(norm, res, out)
    text = report(res)
    (out / "report.md").write_text(text)
    (out / "results.json").write_text(json.dumps(to_json(res), indent=2, default=float))

    print("\n" + text)
    print(f"Overlay: {video}")
    if still:
        print(f"BDC still: {still}")


if __name__ == "__main__":
    main()
