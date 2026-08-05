---
title: Rat Gait Analysis
emoji: 🐀
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Rat Gait / Momentum Analysis Pipeline

Python replacement/extension for the legacy MATLAB DigiGait (Mouse
Specifics, Inc.) workflow. Detects red-marked paw-contact regions in
under-belly treadmill video, tracks all four paws (LF/RF/LH/RH), and
computes gait + momentum metrics with DigiGait-style plots and CSV
export.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Easiest way to run: the web interface

```bash
./run_app.sh
```

(or `./.venv/bin/python server.py`), then open **http://localhost:5050**
in a browser. Pick a video from the list (or drag-and-drop / click to
upload a new one), click **GO**, and see everything -- an annotated
snapshot + looping GIF showing exactly which paw is being tracked
where, all three DigiGait-style plots, the summary metrics table, and
(for videos with DigiGait reference data) a comparison table -- with a
CSV download button. No command-line flags, no GUI popups.

- `config.VIDEO_DIR` (or the `RAT_GAIT_VIDEO_DIR` env var) points at an
  existing local video library, if the machine running the server has
  one. Optional -- on a deployed server this simply won't exist and is
  skipped.
- `uploads/` is where videos uploaded through the browser land. This
  folder is part of the app itself, so it works the same locally and
  once deployed.

## Deploying it (not just localhost)

### Render.com free tier -- quick start

1. Push this repo to GitHub (`git remote add origin <your-repo-url>`,
   `git branch -M main`, `git push -u origin main`).
2. On [render.com](https://render.com): **New +** -> **Blueprint** ->
   connect the repo. Render reads `render.yaml` and configures
   everything automatically (free plan, `requirements-render.txt`,
   gunicorn start command) -- just click **Apply**.
   (No Blueprint option? **New +** -> **Web Service** instead, then set
   Build Command to `pip install -r requirements-render.txt` and Start
   Command to `gunicorn server:app --bind 0.0.0.0:$PORT --workers 1
   --threads 4 --timeout 600`, plan **Free**.)
3. Wait for the build (~3-5 min) -- you get a public URL like
   `https://rat-gait-analysis.onrender.com`.

**Free tier limits to know going in:** the service sleeps after 15 min
with no traffic (next request takes ~30-50s to wake it back up); RAM is
capped at 512MB, so very large videos may process slowly or fail --
watch for that and upgrade to the $7/mo Starter plan (more RAM, less
sleep) if it becomes a problem; and storage is ephemeral -- uploaded
videos and generated results are lost on redeploy/restart, so download
anything you need (the CSV, the plots) before that happens. None of
this is a code problem, it's what "free" costs on any platform for an
app that does real video processing.

This is a plain Flask app (`server.py` + `templates/` + `static/` +
`requirements.txt` + `Procfile`), so it deploys the normal way:

```bash
gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 600
```

(exactly what the included `Procfile` runs). A few things that matter
for THIS app specifically, because the videos are large and analysis is
CPU-heavy:

- **Long timeout.** A real video can take up to a couple of minutes to
  process. Most platforms default to a much shorter request timeout
  (Flask's dev server has none; gunicorn defaults to 30s) -- the
  `Procfile` sets `--timeout 600`, but double-check whatever's in front
  of gunicorn (nginx, the platform's own load balancer) isn't cutting
  requests off sooner.
- **Persistent storage.** `uploads/` and `output/` need to survive
  restarts/redeploys. Several PaaS free tiers use an ephemeral
  filesystem that wipes on every deploy -- if you pick one of those,
  attach a persistent volume/disk, or accept that uploads disappear on
  redeploy.
- **Upload size.** `config.MAX_UPLOAD_BYTES` currently allows up to
  500MB (real DigiGait AVIs run 100-350MB). Some platforms cap request
  body size below that regardless (e.g. a reverse proxy default of
  100MB) -- check your platform's limit.
- **No login/auth.** Anyone with the URL can upload a video and run
  analysis. Fine for a private/internal tool; add basic auth (or put it
  behind your organization's VPN/SSO) before sharing the link widely.
- `opencv-python` in `requirements.txt` pulls in GUI libraries (Qt) that
  a headless Linux server doesn't need and may not have installed by
  default -- swap it for `opencv-python-headless` in `requirements.txt`
  for deployment (keep the GUI version only if you still use `main.py`'s
  CLI manual-calibration flow, which needs it).

## Project layout

| File | Purpose |
|---|---|
| `config.py` | All tunable settings: paths, HSV thresholds, tape width, belt speed override, orientation method |
| `video_io.py` | Belt-speed parsing from filename, dynamic fps/frame reading |
| `calibration.py` | Pixel-to-cm calibration from the 2.5 cm black tape (auto-detect + manual AOI fallback) |
| `orientation.py` | Nose-direction input (manual click or config value) |
| `paw_detection.py` | HSV red segmentation + blob/contour detection per frame |
| `paw_labeling.py` | Quadrant-based LF/RF/LH/RH assignment, builds per-paw time series |
| `metrics.py` | All gait/momentum metrics |
| `plotting.py` | Dynamic Gait Signals, Ensemble Paws, Posture Plot |
| `main.py` | CLI entry point -- orchestrates the full pipeline, writes CSVs |
| `digigait_reference.py` | Auto-loads DigiGait's own calibration/belt-speed/orientation/reference-output when a video has a sibling `_images` folder |
| `paw_overlay.py` | Draws detected paw ellipses + labels on real frames -- annotated snapshot photo + looping GIF |
| `server.py` | Flask backend for the web interface (`templates/`, `static/`) |
| `tools/mask_tuner.py` | Interactive trackbar tool to tune HSV/RGB-ratio thresholds on a real frame |

## Usage

1. Drop raw `.mp4`/`.avi` videos into `videos/`. Filenames should embed
   belt speed, e.g. `a1_day11_24cms.mp4` → 24 cm/s. If a filename can't
   be parsed, the pipeline prompts for it (or set
   `config.BELT_SPEED_OVERRIDE_CMS`).

2. (Recommended) Tune the red-paw and black-tape HSV thresholds against
   a real frame before running the full batch:

   ```bash
   python tools/mask_tuner.py --video videos/a1_day11_24cms.mp4 --target red
   python tools/mask_tuner.py --video videos/a1_day11_24cms.mp4 --target tape
   ```

   Press `p` in the tuner window to print the current values in the
   exact format expected by `config.py`, then paste them in.

3. Run the pipeline:

   ```bash
   python main.py                       # every video in videos/
   python main.py --video videos/x.mp4  # a single video
   ```

   For each video you'll be asked to click the rat's nose position on
   the first frame (orientation) and, if auto-calibration on the tape
   fails, to drag a box across the tape (calibration).

4. Outputs land in `output/plots/*.png` and `output/csv/*.csv`
   (per-video summary, area-time-series, stance-width, and stride
   detail files, plus a combined `all_videos_summary.csv`).

## Paw model: best-fit ellipse + peak-loading instant

This is the part everything else is built on, so it's worth stating
explicitly (mirrors the client's/DigiGait's methodology):

- Every paw, every frame, is modeled as a **best-fit ellipse**
  (`cv2.fitEllipse` on the red contour, in `paw_detection.py`) -- not a
  raw pixel count. Hind paws come out long/narrow (major axis >> minor
  axis); front paws come out closer to circular.
- **Area = ellipse area** (`pi * major/2 * minor/2`), not contour pixel
  area. This is the number that drives Dynamic Gait Signals, dA/dT, and
  stance-phase detection.
- **Peak loading assumption**: within each stance phase, the frame with
  the *maximum* ellipse area is assumed to be the instant of peak leg
  loading.
- **Paw Placement Angle** (angle between the ellipse's long axis and a
  line through the animal, approximated as the direction of travel) is
  computed **only at that peak-loading frame** -- not averaged across
  the stance phase. Same for the paw length/width reported per stance
  phase (`paw_length_at_peak_cm`, `paw_width_at_peak_cm` in the summary
  CSV): both are read at the peak frame, then averaged across stance
  phases for the summary row.

## Validating against a reference DigiGait output

- `output/csv/<video>_summary.csv` has one row per paw with every scalar
  metric (stride length, stance width, paw angle, dA/dT min/max, etc.)
  -- diff this against the reference numbers.
- `output/plots/<video>_dynamic_gait_signals.png`,
  `_ensemble_paws.png`, and `_posture_plot.png` are styled to match the
  DigiGait report layout for visual comparison.
- If left/right paws come out swapped, flip `config.MIRROR_LEFT_RIGHT`
  (the camera looks up at the rat's belly, which mirrors left/right
  relative to a from-above view).
- `step_sequence_regularity_pct` and `animal_length_cm`/`animal_width_cm`
  are documented approximations in `metrics.py` (DigiGait's exact
  formulas aren't public, and we only have paw-contact blobs, not a full
  body silhouette) -- expect to need to retune these two once you have
  reference numbers to check against.

## Tuning notes

- All thresholds live in `config.py`. The paw-labeling logic
  (`paw_labeling.label_blobs_in_frame`) and the red/tape HSV ranges are
  the two most likely things to need adjustment on real footage.
- `config.MIN_PAW_BLOB_AREA_PX` filters sensor noise; raise it if you
  see spurious tiny blobs, lower it if faint/early contact is being
  missed.
- `config.STANCE_AREA_FRACTION_THRESHOLD` controls how much of a paw's
  peak area counts as "in stance" for stride/footfall detection.
