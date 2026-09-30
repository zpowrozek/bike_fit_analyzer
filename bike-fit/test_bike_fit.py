"""Checks the analysis on a simulated rider with known angles (no video or model needed).

The rider walks up to the bike (3 s), pedals seated at 90 rpm (10 s), then coasts (3 s).
Run: python -m pytest test_bike_fit.py  (or: python test_bike_fit.py)
"""

import numpy as np

import bike_fit as bf

FPS = 30.0
PX_PER_MM = 0.5
THIGH = SHANK = 430 * PX_PER_MM
CRANK = 170 * PX_PER_MM
BB = np.array([600.0, 800.0])
SEAT_DIR = np.array([-np.cos(np.radians(73)), -np.sin(np.radians(73))])  # BB -> saddle


def leg_reach(flex_deg):
    th = np.radians(180 - flex_deg)
    return np.sqrt(THIGH**2 + SHANK**2 - 2 * THIGH * SHANK * np.cos(th))


def knee_from(hip, ankle):
    d = ankle - hip
    dist = np.linalg.norm(d)
    u = d / dist
    h = np.sqrt(max(THIGH**2 - (dist / 2) ** 2, 0.0))
    return hip + u * dist / 2 + np.array([-u[1], u[0]]) * -h  # knee points forward (+x)


def simulate(bdc_flex=45.0, torso=45.0, shoulder=88.0, elbow=20.0, seed=0):
    rng = np.random.default_rng(seed)
    hip_seated = BB + SEAT_DIR * (leg_reach(bdc_flex) - CRANK)
    n = int(16 * FPS)
    xy = rng.normal(0, 30, (n, 17, 2)) + 400  # far-side junk
    conf = np.full((n, 17), 0.35)
    near = bf.SIDES["right"]
    for i in range(n):
        t = i / FPS
        if t < 3:  # walking up to the bike: hip travels, legs swing
            hip = np.array([200 + 100 * t, hip_seated[1] - 40])
            swing = np.radians(20 * np.sin(2 * np.pi * 1.0 * t))
            ankle = hip + (THIGH + SHANK - 5) * np.array([np.sin(swing), np.cos(swing)])
        else:
            hip = hip_seated.copy()
            crank = 2 * np.pi * 1.5 * (t - 3) if t < 13 else 0.0  # 90 rpm, then coast at 3 o'clock
            ankle = BB + CRANK * np.array([np.cos(crank), np.sin(crank)])
        knee = knee_from(hip, ankle)
        sh = hip + 300 * np.array([np.cos(np.radians(torso)), -np.sin(np.radians(torso))])
        up = np.radians(135 - shoulder)
        el = sh + 150 * np.array([np.cos(up), np.sin(up)])
        fo = up - np.radians(elbow)
        wr = el + 140 * np.array([np.cos(fo), np.sin(fo)])
        for name, p in dict(shoulder=sh, elbow=el, wrist=wr, hip=hip, knee=knee, ankle=ankle).items():
            xy[i, near[name]] = p + rng.normal(0, 1.0, 2)
            conf[i, near[name]] = 0.9
    # Motion blur: a few dropped knees and one wild ankle.
    conf[rng.choice(n, n // 30, replace=False), near["knee"]] = 0.2
    xy[200, near["ankle"]] += 60
    return xy.astype(np.float32), conf.astype(np.float32)


def test_finds_strokes_only_while_pedaling():
    xy, conf = simulate()
    res = bf.analyze(xy, conf, FPS, height_cm=175)
    assert res["side"] == "right"
    bdc_s = np.array(res["bdc"]) / FPS
    assert bdc_s.min() > 3 and bdc_s.max() < 13  # no walking or coasting frames
    assert 12 <= len(res["bdc"]) <= 16  # ~15 strokes in 10 s at 90 rpm
    assert abs(res["cadence"] - 90) < 3
    assert not res["pedaling"][: int(3 * FPS)].any()
    assert not res["pedaling"][int(13.5 * FPS):].any()


def test_measures_known_angles():
    xy, conf = simulate(bdc_flex=45, torso=45, shoulder=88, elbow=20)
    m = bf.analyze(xy, conf, FPS, height_cm=175)["metrics"]
    assert abs(m["knee"]["value"] - 45) < 1.0
    assert abs(m["torso"]["value"] - 45) < 1.0
    assert abs(m["shoulder"]["value"] - 88) < 1.0
    assert abs(m["elbow"]["value"] - 20) < 1.0
    assert m["knee"]["grade"] == "red"
    assert [m[k]["grade"] for k in ("torso", "shoulder", "elbow")] == ["green"] * 3


def test_saddle_advice_moves_knee_into_green():
    xy, conf = simulate(bdc_flex=45)
    res = bf.analyze(xy, conf, FPS, height_cm=175)
    saddle = res["advice"][0]
    assert "raise" in saddle["text"] and saddle["grade"] == "red"
    # Apply the exact (unrounded) change to the simulated bike: the knee should land at ~38.5.
    mm = bf.saddle_change_mm(res["metrics"]["knee"]["value"], 38.5, res["thigh_mm"], res["shank_mm"])
    raised = leg_reach(45) + mm * PX_PER_MM
    flex = 180 - np.degrees(np.arccos((THIGH**2 + SHANK**2 - raised**2) / (2 * THIGH * SHANK)))
    assert abs(flex - 38.5) < 1.0


def test_hip_graded_at_bottom_not_top():
    xy, conf = simulate(bdc_flex=38, torso=45)
    res = bf.analyze(xy, conf, FPS, height_cm=175)
    assert res["metrics"]["hip"]["grade"] == "green"  # ~99 deg at the bottom
    assert res["hip_top"] < 70  # the top closes far below 85-110, so it's not graded


def test_grading_edges():
    assert bf.grade(35, 30, 40) == "green"
    assert bf.grade(42, 30, 40) == "amber"
    assert bf.grade(43, 30, 40) == "red"
    assert bf.grade(28, 30, 40) == "amber"
    assert bf.grade(27, 30, 40) == "red"


def test_closed_cockpit_colors_arm_red():
    xy, conf = simulate(bdc_flex=35, shoulder=70, elbow=20)
    res = bf.analyze(xy, conf, FPS, height_cm=175)
    assert res["metrics"]["knee"]["grade"] == "green"
    assert res["colors"]["arm"] == "red"
    assert "longer stem" in res["advice"][1]["text"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
