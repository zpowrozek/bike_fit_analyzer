# Bike-fit analyzer

Grades a road-bike fit from a short side-on phone video of you pedaling on a trainer.
It measures your joint angles at the bottom of the pedal stroke, grades them
green / amber / red, draws a colored skeleton over the clip, and writes a short
report with the fix (e.g. "knee 45 deg at the bottom -> saddle too low, raise ~15 mm").

## Run

```
pip install -r requirements.txt
python bike_fit.py path/to/video.mov --height-cm 180
```

The first run downloads the `yolo11x-pose.pt` model (~113 MB). Results go to
`results/<video name>/`:

| File | What |
|---|---|
| `overlay.mp4` | whole clip with the colored skeleton (H.264, plays anywhere) |
| `bdc_still.jpg` | a representative bottom-of-stroke frame |
| `report.md` | angles, grades and what to change, in order |
| `results.json` | the same numbers, machine readable |

Tests (simulated rider, no video needed): `python -m pytest test_bike_fit.py`

## Filming

- Phone level, at hip height, square to the bike, 2-3 m away; landscape; whole body and bike in frame.
- Warm up, hands on the hoods, then 15-20 s of steady seated pedaling (~85-95 rpm).
- Fitted clothing and good light help the model find your joints.

## Grading (dynamic, road, measured while pedaling)

| Angle | Green | Meaning if off |
|---|---|---|
| Knee bend at bottom | 30-40 deg | over ~42 saddle too low; under ~28 saddle too high |
| Torso from horizontal | 40-50 deg | lower = aggressive front, higher = upright |
| Elbow bend | 15-30 deg | under = arms locked (reach long) |
| Shoulder (torso to upper arm) | 80-95 deg | under = closed cockpit (reach short / weight on hands) |
| Hip (torso to thigh) at bottom | 85-110 deg | info only, flexibility dependent |

Amber = within 2.5 deg of green (a phone video resolves angles to a few degrees).

Sources: the knee zone is the Holmes method's static 25-35 deg plus the ~8 deg that
dynamic video adds (Bini et al. 2014, static knee 8.2 +/- 5 deg smaller at 6 o'clock,
PubMed 24499342); field averages knee 36 +/- 7 and elbow 19 +/- 8 deg (PMC9219349);
torso/shoulder/hip from bikefitadviser.com "Not basic bike fit, part 3".

The 85-110 deg hip range matches the hip at the *bottom* of the stroke. At the top a
road rider's hip closes to roughly 45-60 deg, so that number is reported but not graded.

## How it works

1. ffmpeg normalizes the video (applies phone rotation, H.264, max 60 fps / 1920 px).
2. YOLO11x-pose finds 17 COCO keypoints per frame; the most confident person and the
   camera-side shoulder/elbow/wrist/hip/knee/ankle are kept (points under 0.5
   confidence are ignored, as are frames where a limb segment's length jumps).
3. Pedaling = knee swinging 25+ deg in a 2 s window while the hip stays planted, which
   drops mounting, walking around and coasting.
4. Bottom-dead-center is the knee's most-extended frame on each stroke; angles are
   median-ed across strokes.
5. Saddle mm estimate: the hip-to-ankle distance change that moves the knee back just
   inside green, using your leg proportions from the video scaled by your height.

Saddle setback (fore-aft) isn't measured. Change one thing at a time and re-film.

## Licenses

ultralytics is AGPL-3.0; supervision is MIT.
