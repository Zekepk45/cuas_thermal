# Team Thermal – Boson thermal tracker (block A4)

Runs the **Thermal Processing (A4)** step from the SRR block diagram on live
Boson+ 640 video:

```
grab frame → find hot target → sub-pixel centroid → Kalman track → az/el angle
          → timestamp + health check → UDP JSON to fusion (A6) + CSV log (DL1)
```

No gimbal yet: angles are measured from the camera's boresight (the white
crosshair). Fusion will later add gimbal angles and the surveyed pod heading.

## Files
| File | What it does |
|---|---|
| `run_tracker.py` | Main program. Opens the stream, detects, tracks, displays, sends, logs. |
| `detector.py` | Hot-spot detector (spatial top-hat + static-clutter suppression + noise threshold). |
| `tracker.py` | Single-target constant-velocity Kalman filter with SEARCH / TENTATIVE / TRACK / COAST logic. |
| `camera_model.py` | Pixel → az/el and LOS unit vector. Lens profiles for the Seeker (4.9 mm) and pod (50 mm) Bosons. |
| `output.py` | `thermal_los` UDP JSON message and CSV logger. |
| `udp_listen.py` | Stand-in fusion node: prints incoming messages, rate (Hz) and age (ms). |
| `make_test_video.py`, `score_run.py`, `test_sky.mp4` | Synthetic test video with truth, and a scorer. |

## Install (laptop)
```
pip install opencv-python numpy
```
(Needs Python 3.8+. `opencv-python` includes FFmpeg, which is what reads RTSP.)

## Run it on the Seeker drone's Boson (4 terminals)

**T1 – start the stream on the drone** (props off, battery in, USB-C to laptop):
```
adb shell voxl-streamer -i <your boson pipe name>
```
**T2 – forward the stream over USB:**
```
adb forward tcp:8900 tcp:8900
```
**T3 – fake fusion node (shows what fusion would receive):**
```
python3 udp_listen.py
```
**T4 – the tracker:**
```
you need to create a virtual environment to run the next line
run:
source .venv/bin/activate
then:
python3 run_tracker.py --source rtsp://127.0.0.1:8900/live --lens seeker-4.9mm
```

Window controls: **click** a hot spot to lock onto it · **r** reset · **c** clear lock ·
**s** snapshot · **q** quit. Box colors: green = TRACK (valid), yellow = TENTATIVE,
orange = COAST (lost it briefly, predicting).

Every run writes `logs/thermal_BOSON-SEEKER_<time>.csv`. Add `--record run1.mp4`
to save the annotated video for the brief.

### Which mode to use
| Situation | Command add-on |
|---|---|
| Drone flying outside, small/far, sky + trees/buildings | *(default)* `--mode local` |
| Lock onto **only the hottest object** in view (hotplate, heat beacon) | `--mode hottest` |
| Bench demo: hand, mug, soldering iron or drone 1–10 m away | `--mode global` |
| Camera on a moving gimbal (later) | `--static-tau 0` |
| Too many false hits | raise `--k-sigma` (e.g. 6–7) |
| Missing a faint target | lower `--k-sigma` (e.g. 4) |
| Image looks garbled / smeared | `--no-low-latency` |
| Stream is black-hot | `--black-hot` |

`hottest` picks the single hottest blob in every frame, whatever its size, and does
not fade out objects that sit still. `--hot-delta` (default 12 gray levels) sets how far
below the hottest pixel still counts as part of the object.

`local` keeps only warm objects smaller than `--bg-kernel` pixels (default 15),
which is what a drone at range looks like, and ignores big warm areas such as
roofs and the tree line. A hand at 1 m is much bigger than that, so use
`global` on the bench.

Hot spots that **don't move** fade out after about `--static-tau` seconds (default 4).
That removes warm clutter, but the camera must stay still. The area around a
target that is already being tracked is protected, so a hovering drone stays
in track.

## Output to fusion: `thermal_los` message (UDP JSON)
Default destination is `127.0.0.1:9101` (`--udp-host`, `--udp-port`). Last year's
SkyDome radar feed used 9001; this keeps the same UDP-JSON style on a separate port.
One message is sent **every frame** (30–60 Hz, well above the ≥5 Hz KSA), valid or not,
so fusion always knows the sensor's status.

```json
{"msg_type":"thermal_los","schema":1,"sensor_id":"BOSON-SEEKER","seq":1234,
 "t_unix":1792003026.071712,"t_iso":"2026-10-14T18:37:06.071712Z","time_source":"host_rx",
 "track_state":"TRACK","track_id":3,"valid":true,"fault":null,
 "lens":"seeker-4.9mm","frame_wh":[640,512],"fps":30.0,
 "az_deg":4.07912,"el_deg":17.47401,"los_frd":[0.9516,0.0679,-0.2999],
 "sigma_deg":0.0805,"az_rate_dps":2.91,"el_rate_dps":0.84,
 "px":[347.12,126.01],"snr":20.4,"area_px":9}
```
| Field | Meaning |
|---|---|
| `az_deg`, `el_deg` | Angle from boresight. az + right, el + up. |
| `los_frd` | Unit line-of-sight vector in camera Forward-Right-Down. Fusion rotates this into NED. |
| `sigma_deg` | 1-sigma angle uncertainty from the Kalman filter (goes into fusion's R matrix). |
| `valid` | `true` only when the track is confirmed (TRACK) and no fault. Fusion should ignore invalid messages (hold path). |
| `track_state` | SEARCH / TENTATIVE / TRACK / COAST. |
| `fault` | `null`, `no_video`, `frame_gap_X.XXs`, `timestamp_not_increasing` (fault path → hold). |
| `time_source` | `host_rx` = laptop clock when the frame arrived (includes stream latency); `file_fps` = recorded video. |

## Test without the drone
```
python3 run_tracker.py --source test_sky.mp4            # watch it track
python3 run_tracker.py --source test_sky.mp4 --no-display
python3 score_run.py logs/<newest>.csv test_sky_truth.csv
```
Result on the included synthetic video (2.5 px target, tree line, warm roof, 0.4 s dropout):
99 % of visible frames tracked, 0 wrong-target frames, 0.2 px median centroid error,
~10 ms per frame. During the dropout the track coasted (valid = false) and reacquired
with the same track ID.

## Known limits (be ready to say these at the brief)
1. **Lens:** the Seeker's 4.9 mm lens is about 2.45 mrad/pixel (~76° HFOV). A 0.35 m drone is
   ~2.9 px at 50 m and ~1.4 px at 100 m. Use it to prove the pipeline, not the range. The
   pod's 50 mm lens is ~0.24 mrad/pixel: the same drone is ~4.8 px at 300 m.
2. **Compression:** the RTSP stream is H.264. In testing, compression reduced a faint
   1–2 px target's contrast by 3–7×, below what can be detected. Close/bright targets
   are fine. For range testing, increase the streamer bitrate in
   `/etc/modalai/voxl-streamer.conf` or use raw frames.
3. **Timestamps:** `host_rx` time includes stream and decode latency (typically tens to
   hundreds of ms). Measure it once: film the laptop's clock with a phone while you
   swing a hot object into view, and compare that time with when the log shows the
   track starting. (A thermal camera can't see a phone screen.) The pod Bosons
   on USB will have much lower latency. Final timing comes from the GPS/PTP time reference (TS1).
4. **Distortion:** the pinhole model is approximate near the edges of the 4.9 mm lens.
   Calibrate (OpenCV checkerboard heated or cooled) before using the angles for scoring.
5. **Single target:** the tracker follows one target (the best one or the one you click).
