"""
Loader for DigiGait's own per-video metadata and reference output.

Some videos (in this project, the ones that were already run through
DigiGait) have a sibling "<video_stem>_images/" folder next to the raw
.avi, produced by DigiGait itself. When it exists, it contains exact
ground-truth values we'd otherwise have to guess or ask the user for:

  AOI_dimensions.txt                    -> exact px<->cm calibration
  belt_speed.txt                        -> exact belt speed
  gait_result_mouse_mouth_coordinates.xls -> exact per-frame snout position
                                              (used to derive orientation,
                                              skipping the manual click)
  result/INDICES_<stem>.xls             -> DigiGait's own computed metrics
                                              per paw -- for validation
  result/PawArea_<stem>.xls             -> DigiGait's own per-frame paw
                                              area (px^2) -- for validation

All of this is OPTIONAL: main.py checks whether the folder exists and
falls back to the interactive/manual flow (calibration.py, orientation.py,
manual belt-speed entry) when it doesn't -- most raw videos in a batch
won't have been pre-processed by DigiGait.

NOTE: RGB_Values_for_this_mouse_paw.txt (also found in these folders)
looked like it should give an exact per-animal paw-color threshold, but
its 4 numbers didn't reproduce anything sensible when tried as literal
R/G, R/B ratio thresholds against the real frame (max observed R/G in a
real sample frame was ~1.68, but one of the numbers is a 2.0 ratio floor
-- it matched zero pixels). We could not reverse-engineer its exact
meaning with confidence, so it's intentionally NOT used here. Paw-color
detection instead uses the empirically-tuned default in config.py
(see paw_detection.build_red_mask_rgb_ratio), validated by comparing
resulting blob positions against a real bw_image_*.jpg ground-truth mask.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from calibration import Calibration
from orientation import Orientation, direction_from_point

LABEL_TO_LIMB = {"LF": "Left Fore", "RF": "Right Fore", "LH": "Left Hind", "RH": "Right Hind"}


def find_images_dir(video_path: Path) -> Path | None:
    """DigiGait's own per-video output folder, if it exists next to the video."""
    candidate = video_path.parent / f"{video_path.stem}_images"
    return candidate if candidate.is_dir() else None


def _read_first_number_row(path: Path) -> list[float]:
    """Parse the first whitespace/tab-delimited numeric line of a small text file."""
    with open(path) as f:
        line = f.readline()
    return [float(x) for x in line.split()]


def load_aoi_calibration(images_dir: Path) -> Calibration | None:
    """
    AOI_dimensions.txt has one row: width_cm, height_cm, width_px, height_px.
    On real sample data this comes out anisotropic (non-square pixels from
    the original analog capture) -- Calibration.from_aoi_dimensions keeps
    x/y scale separate rather than assuming a single ratio.
    """
    path = images_dir / "AOI_dimensions.txt"
    if not path.exists():
        return None
    try:
        width_cm, height_cm, width_px, height_px = _read_first_number_row(path)
    except (ValueError, OSError):
        return None
    return Calibration.from_aoi_dimensions(width_cm, height_cm, width_px, height_px)


def load_belt_speed(images_dir: Path) -> float | None:
    path = images_dir / "belt_speed.txt"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (ValueError, OSError):
        return None


def load_nose_orientation(images_dir: Path, aoi_width_px: float, aoi_height_px: float) -> Orientation | None:
    """
    gait_result_mouse_mouth_coordinates.xls has one (X, Y) snout pixel
    coordinate per frame (tab-separated text despite the .xls name). We
    use the first several frames' median position (more robust than a
    single frame to any one-off detection glitch) and derive a cardinal
    direction the same way a manual nose click would be interpreted.
    """
    path = images_dir / "gait_result_mouse_mouth_coordinates.xls"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t")
    except (OSError, pd.errors.ParserError):
        return None
    if df.shape[0] == 0 or df.shape[1] < 2:
        return None

    df.columns = [c.strip() for c in df.columns]
    xs = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    ys = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
    if xs.empty or ys.empty:
        return None

    n = min(20, len(xs))
    snout_x = float(xs.iloc[:n].median())
    snout_y = float(ys.iloc[:n].median())

    direction = direction_from_point(snout_x, snout_y, aoi_width_px, aoi_height_px)
    return Orientation(nose_direction=direction, nose_point_px=(snout_x, snout_y))


@dataclass
class DigiGaitMetadata:
    images_dir: Path | None
    calibration: Calibration | None
    belt_speed_cms: float | None
    orientation: Orientation | None
    # True when this metadata was borrowed from OTHER videos in the same
    # folder (see find_session_defaults), not this video's own DigiGait
    # output -- callers should skip reference-comparison lookups in that
    # case, since there's no per-video INDICES/PawArea file to compare against.
    is_session_default: bool = False


def _load_metadata_from_images_dir(images_dir: Path) -> DigiGaitMetadata | None:
    calibration = load_aoi_calibration(images_dir)
    belt_speed = load_belt_speed(images_dir)

    orientation = None
    if calibration is not None:
        try:
            # AOI pixel dims double as the frame dims the snout
            # coordinates were recorded against.
            _, _, aoi_w_px, aoi_h_px = _read_first_number_row(images_dir / "AOI_dimensions.txt")
            orientation = load_nose_orientation(images_dir, aoi_w_px, aoi_h_px)
        except (ValueError, OSError):
            orientation = None

    if calibration is None and belt_speed is None and orientation is None:
        return None

    return DigiGaitMetadata(
        images_dir=images_dir,
        calibration=calibration,
        belt_speed_cms=belt_speed,
        orientation=orientation,
    )


def load_metadata(video_path: Path) -> DigiGaitMetadata | None:
    """Convenience: load everything auto-derivable for one video, or None
    if it has no matching DigiGait '_images' output folder."""
    images_dir = find_images_dir(video_path)
    if images_dir is None:
        return None
    return _load_metadata_from_images_dir(images_dir)


def find_session_defaults(video_dir: Path) -> DigiGaitMetadata | None:
    """
    Fallback for videos with no '_images' output folder of their own:
    borrow calibration/orientation from OTHER videos in the same
    directory, if every one of them agrees. Real footage does NOT always
    show the calibration tape in-frame (the AOI/calibration is set up
    once per recording session in DigiGait, not re-derived per video),
    so per-video auto-detection can fail even though the whole session
    genuinely shares one calibration.

    Verified on the real dataset this was built against: AOI_dimensions.txt
    was byte-identical and orientation resolved identically across every
    already-processed video in the same session folder -- so this is a
    justified default for videos from the same rig/session, not a blind
    guess. If the candidates disagree, returns None rather than picking
    one arbitrarily.
    """
    candidates = []
    for images_dir in sorted(video_dir.glob("*_images")):
        if not images_dir.is_dir():
            continue
        meta = _load_metadata_from_images_dir(images_dir)
        if meta is not None and meta.calibration is not None:
            candidates.append(meta)

    if not candidates:
        return None

    first_cal = candidates[0].calibration
    for meta in candidates[1:]:
        cal = meta.calibration
        if (abs(cal.cm_per_pixel_x - first_cal.cm_per_pixel_x) > 1e-4
                or abs(cal.cm_per_pixel_y - first_cal.cm_per_pixel_y) > 1e-4):
            return None  # disagreement -- don't guess which one applies

    orientations = {m.orientation.nose_direction for m in candidates if m.orientation is not None}
    shared_orientation = (
        Orientation(nose_direction=next(iter(orientations))) if len(orientations) == 1 else None
    )

    return DigiGaitMetadata(
        images_dir=None,
        calibration=first_cal,
        belt_speed_cms=None,  # belt speed still comes from this video's OWN filename
        orientation=shared_orientation,
        is_session_default=True,
    )


# ---------------------------------------------------------------------------
# Reference output (for validation, not for driving the pipeline)
# ---------------------------------------------------------------------------

def load_reference_paw_area(images_dir: Path, video_stem: str, calibration: Calibration) -> pd.DataFrame | None:
    """
    result/PawArea_<stem>.xls: DigiGait's own per-frame paw area in px^2,
    columns "Left Fore Area", "Right Fore Area", "Left Hind Area",
    "Right Hind Area". Converted to cm^2 and renamed to our LF/RF/LH/RH
    convention so it lines up directly with our own area_time_series.
    """
    path = images_dir / "result" / f"PawArea_{video_stem}.xls"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t")
    except (OSError, pd.errors.ParserError):
        return None
    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "Left Fore Area": "LF", "Right Fore Area": "RF",
        "Left Hind Area": "LH", "Right Hind Area": "RH",
    }
    df = df.rename(columns=rename)
    for col in ("LF", "RF", "LH", "RH"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * calibration.cm_per_pixel_x * calibration.cm_per_pixel_y
    return df


def load_reference_indices(images_dir: Path, video_stem: str) -> pd.DataFrame | None:
    """
    result/INDICES_<stem>.xls: DigiGait's own computed summary metrics,
    one row per limb (fore file + hind file are actually concatenated
    into one file with a 'Limb' column). Skips the units row.
    """
    fore_path = images_dir / "result" / f"INDICES_{video_stem}.xls"
    if not fore_path.exists():
        return None
    try:
        df = pd.read_csv(fore_path, sep="\t", skiprows=[1])
    except (OSError, pd.errors.ParserError):
        return None
    df.columns = [c.strip() for c in df.columns]
    if "Limb" in df.columns:
        df["Limb"] = df["Limb"].astype(str).str.strip()
    return df


def compare_summary_to_reference(our_summary_df: pd.DataFrame, indices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side comparison: our computed summary metrics vs. DigiGait's
    own INDICES values, per paw, for the metrics both sides compute.
    """
    field_map = [
        ("mean_stride_length_cm", "StrideLength"),
        ("stance_width_cm", "StanceWidth"),
        ("mean_step_angle_deg", "StepAngle"),
        ("paw_placement_angle_deg", "PawAngle"),
        ("dadt_max_cm2_s", "MAX dA/dT"),
        ("dadt_min_cm2_s", "MIN dA/dT"),
        ("peak_paw_area_cm2", "Paw Area at Peak Stance in sq. cm"),
    ]

    rows = []
    for _, our_row in our_summary_df.iterrows():
        label = our_row["paw"]
        limb = LABEL_TO_LIMB.get(label)
        ref_rows = indices_df[indices_df["Limb"] == limb] if "Limb" in indices_df.columns else pd.DataFrame()
        if ref_rows.empty:
            continue
        ref_row = ref_rows.iloc[0]

        for our_col, ref_col in field_map:
            if our_col not in our_row or ref_col not in indices_df.columns:
                continue
            our_val = our_row[our_col]
            ref_val = pd.to_numeric(pd.Series([ref_row[ref_col]]), errors="coerce").iloc[0]
            pct_diff = (
                100.0 * (our_val - ref_val) / abs(ref_val)
                if pd.notna(ref_val) and ref_val != 0 and pd.notna(our_val)
                else float("nan")
            )
            rows.append({
                "paw": label,
                "metric": our_col,
                "our_value": our_val,
                "digigait_value": ref_val,
                "pct_diff": pct_diff,
            })

    return pd.DataFrame(rows)
