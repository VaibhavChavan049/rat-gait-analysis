"""
Central configuration for the rat gait / momentum analysis pipeline.

Everything the user is likely to need to tune (paths, thresholds,
calibration constants, orientation method) lives here so the rest of
the codebase never hardcodes a "magic number".
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Local video library (optional). Only meaningful when running on the
# machine that actually has this folder -- on a deployed server this
# path won't exist, so the app just skips it and relies on UPLOADS_DIR
# instead. Override with the RAT_GAIT_VIDEO_DIR env var if needed.
_video_dir_env = os.environ.get("RAT_GAIT_VIDEO_DIR")
VIDEO_DIR = Path(_video_dir_env) if _video_dir_env else Path(
    "/Users/vaibhavchavan/Documents/Mathworks/Anterios/day11_Nov7"
)

# Videos uploaded through the web interface always land here -- this
# folder is part of the deployed app itself, so it exists everywhere
# (local machine or a server), unlike VIDEO_DIR.
UPLOADS_DIR = PROJECT_ROOT / "uploads"

OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv"

for _d in (UPLOADS_DIR, OUTPUT_DIR, PLOTS_DIR, CSV_DIR):
    _d.mkdir(parents=True, exist_ok=True)
if VIDEO_DIR.is_dir():
    pass  # local library exists on this machine -- nothing to create
else:
    VIDEO_DIR = None  # not available here (e.g. deployed server); upload-only mode

# Largest video upload accepted by the web interface, in bytes. Raw
# DigiGait AVIs run 100-350MB in this dataset -- keep this generous, but
# note that some hosting platforms cap request body size lower than
# this regardless (see README's deployment section).
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

# ---------------------------------------------------------------------------
# Belt speed
# ---------------------------------------------------------------------------
# Belt speed (cm/s) is expected to be embedded in the filename, e.g.
#   "a1_day11_24cms.mp4"  ->  24 cm/s
# Regex used to pull it out lives in video_io.py. If parsing fails, this
# override is used instead (set to None to force manual entry via prompt).
BELT_SPEED_OVERRIDE_CMS = None  # e.g. 24.0

# ---------------------------------------------------------------------------
# Calibration (pixel -> cm)
# ---------------------------------------------------------------------------
# Width of the black calibration tape bordering the treadmill AOI, in cm.
CALIBRATION_TAPE_WIDTH_CM = 2.5

# HSV range used to auto-detect the black tape. Black tape = low value (V),
# any hue/saturation. Tune with tools/mask_tuner.py if auto-detect fails.
BLACK_TAPE_HSV_LOWER = (0, 0, 0)
BLACK_TAPE_HSV_UPPER = (180, 255, 60)

# If auto-detection of the tape fails (or the user wants to be certain),
# fall back to manual AOI selection (click-drag rectangle over the tape).
CALIBRATION_ALLOW_MANUAL_FALLBACK = True

# ---------------------------------------------------------------------------
# Paw (red marker) detection
# ---------------------------------------------------------------------------
# Two detection methods, see paw_detection.py:
#   "hsv":       bright saturated-red thresholding. Good for footage
#                where paw contact is drawn in with a vivid marker color.
#   "rgb_ratio": subtler pink/salmon redness (R channel elevated relative
#                to G). This is what real DigiGait footage actually looks
#                like -- paw contact is a light pressure-induced skin
#                discoloration, not a bright paint overlay. Bright-red HSV
#                thresholds tuned for a marker will miss it almost
#                entirely. Default here because that's the real footage
#                this pipeline was validated against; switch to "hsv" if
#                your videos really do have a vivid marker color.
PAW_DETECTION_METHOD = "rgb_ratio"  # "hsv" | "rgb_ratio"

# Red wraps around the HSV hue circle (0 and 180), so two ranges are used
# and their masks are OR'd together. Tune with tools/mask_tuner.py.
RED_HSV_LOWER_1 = (0, 70, 50)
RED_HSV_UPPER_1 = (10, 255, 255)
RED_HSV_LOWER_2 = (170, 70, 50)
RED_HSV_UPPER_2 = (180, 255, 255)

# rgb_ratio method thresholds. RGB_RATIO_RG_THRESHOLD: minimum R/G ratio
# to count as paw-contact redness. Tuned against the real
# a1_day11_24cms.avi video (which has a DigiGait reference to check
# against) by sweeping thresholds and picking the one whose stance duty
# cycle (% of frames each paw shows nonzero area) best matches DigiGait's
# own reported %StanceStride (~62-69% in the reference INDICES file).
# 1.15 (the first guess, from a single-frame blob-position check) turned
# out far too lenient -- it left paws "in contact" ~99% of frames, which
# collapsed swing phases into stance and badly inflated stride length.
# There's real per-paw variation (1.35 undershoots RF, overshoots LH/RH),
# so this single global value is a compromise -- retune per dataset with
# tools/mask_tuner.py --target rgb_ratio if your footage differs.
# RGB_RATIO_MIN_RED: minimum red-channel brightness, filters dark noise.
RGB_RATIO_RG_THRESHOLD = 1.35
RGB_RATIO_MIN_RED = 87  # reused from RGB_Values_for_this_mouse_paw.txt's
                        # 3rd number, which held up as a plausible min-brightness floor

# Minimum blob area in pixels^2 to be considered a real paw contact
# (filters out sensor noise / small red speckles). Tune per-video if the
# camera resolution / rat size changes a lot.
MIN_PAW_BLOB_AREA_PX = 40

# Morphological kernel size used to clean up the paw mask (close small
# gaps, remove speckle) before contour detection.
MORPH_KERNEL_SIZE = 5

# Maximum number of paw blobs to keep per frame (rat has 4 paws).
MAX_PAWS_PER_FRAME = 4

# ---------------------------------------------------------------------------
# Orientation (nose direction)
# ---------------------------------------------------------------------------
# "manual_click": user clicks the nose position on a sample frame when the
#                 pipeline runs (replicates DigiGait's GUI step).
# "config":       use ORIENTATION_CONFIG below (no interactive prompt) --
#                 useful for batch/headless runs once you know the layout.
ORIENTATION_METHOD = "manual_click"  # "manual_click" | "config"

# Used only when ORIENTATION_METHOD == "config".
# "nose_direction": which way the nose points in the frame.
#   "up", "down", "left", "right"
ORIENTATION_CONFIG = {
    "nose_direction": "up",
}

# The camera looks UP at the rat's underside (ventral view). Relative to a
# dorsal (from-above) view, this mirrors left/right. Leave True by default;
# flip during validation against the reference DigiGait output if LF/RF (or
# LH/RH) come out swapped.
MIRROR_LEFT_RIGHT = True

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
# Max distance (cm) a paw centroid may move between consecutive frames and
# still be considered the "same" paw track (used to keep stable per-paw
# identity across frames independent of the quadrant relabeling).
MAX_TRACK_JUMP_CM = 3.0

# A "stride"/step is defined by contiguous frames where a given paw's
# contact area is above this fraction of that paw's own max observed area.
STANCE_AREA_FRACTION_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
DIGIGAIT_BG_COLOR = "#1a5c3a"     # dark green background used by DigiGait plots
DIGIGAIT_LINE_COLORS = {
    "LF": "#ffff00",  # yellow
    "RF": "#00ffff",  # cyan
    "LH": "#ff00ff",  # magenta
    "RH": "#ffffff",  # white
}
PAW_ORDER = ["LF", "RF", "LH", "RH"]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
VERBOSE = True
