"""
Gait / momentum metrics computed from per-paw time series produced by
paw_labeling.build_paw_tracks().

Each public compute_* function takes the `tracks` dict (label -> list of
PawFrameSample) plus whatever else it needs (fps, belt speed, calibration,
orientation) and returns plain Python / numpy / pandas structures -- no
plotting or I/O happens here, so this module can be unit tested and
reused independently of plotting.py.

NOTE on "animal length/width": we only have paw-contact blobs to work
with (no full body/silhouette segmentation), so these two metrics are
necessarily *proxies* derived from paw spread (fore-vs-hind and
left-vs-right distances), not a true nose-to-tail body measurement. Flag
this clearly when comparing against the reference DigiGait output, and
extend with real body-silhouette detection later if the numbers don't
line up.

NOTE on "step angle": the exact DigiGait formula is proprietary, so this
implements a standard, defensible turning-angle definition (the angle
between successive placement vectors of a paw's own footfall trajectory).
Validate against the reference output and adjust if needed -- that's the
whole point of keeping this in one small, swappable function.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from orientation import Orientation
from paw_labeling import LABELS, PawFrameSample
from video_io import VideoMeta


# ---------------------------------------------------------------------------
# Basic per-frame series
# ---------------------------------------------------------------------------

def area_time_series_dataframe(tracks: dict[str, list[PawFrameSample]]) -> pd.DataFrame:
    """Paw contact area (cm^2) vs time (s), one column per paw -- 'Dynamic Gait Signals' data."""
    time_s = [s.time_s for s in tracks[LABELS[0]]]
    data = {"time_s": time_s}
    for label in LABELS:
        data[label] = [s.area_cm2 if s.present else 0.0 for s in tracks[label]]
    return pd.DataFrame(data)


def compute_dadt(tracks: dict[str, list[PawFrameSample]]) -> dict[str, np.ndarray]:
    """Rate of paw-contact-area change (cm^2/s) per paw -- the momentum proxy."""
    dadt = {}
    for label in LABELS:
        samples = tracks[label]
        t = np.array([s.time_s for s in samples])
        a = np.array([s.area_cm2 if s.present else 0.0 for s in samples])
        dadt[label] = np.gradient(a, t) if len(t) > 1 else np.zeros_like(a)
    return dadt


def dadt_min_max(dadt: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """MIN/MAX dA/dt per paw."""
    return {label: (float(np.min(v)), float(np.max(v))) if len(v) else (0.0, 0.0)
            for label, v in dadt.items()}


# ---------------------------------------------------------------------------
# Stance-phase / footfall detection
# ---------------------------------------------------------------------------

@dataclass
class StanceRun:
    start_idx: int
    end_idx: int
    start_time_s: float
    end_time_s: float
    peak_area_cm2: float
    peak_time_s: float                  # time of max area = the assumed peak-loading instant
    centroid_cm: tuple[float, float]    # mean paw position over the whole stance phase
    peak_length_cm: float               # ellipse major axis, AT the peak-loading frame only
    peak_width_cm: float                # ellipse minor axis, AT the peak-loading frame only
    paw_placement_angle_deg: float      # ellipse major-axis angle vs. body line, AT the peak-loading frame only


def find_stance_runs(
    samples: list[PawFrameSample],
    area_fraction_threshold: float = config.STANCE_AREA_FRACTION_THRESHOLD,
) -> list[StanceRun]:
    """
    Find contiguous stance phases: runs of frames where contact area is at
    least `area_fraction_threshold` of that paw's own max observed area
    (filters out marginal/noisy edge-of-contact frames).

    Per the client's paw model: the frame within each stance phase where
    ellipse area is at its MAXIMUM is assumed to be the instant of peak
    leg loading (the paw is most fully flat against the belt). Paw
    length/width/placement-angle are therefore taken ONLY from that one
    peak frame -- not averaged across the whole stance phase -- because
    that's specifically the instant those measurements are meaningful for.
    """
    areas = np.array([s.area_cm2 if s.present else 0.0 for s in samples])
    max_area = float(areas.max()) if areas.size else 0.0
    if max_area <= 0:
        return []

    threshold = area_fraction_threshold * max_area
    is_stance = areas >= threshold

    runs = []
    start = None
    for i, flag in enumerate(is_stance):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_stance) - 1))

    stance_runs = []
    for start_idx, end_idx in runs:
        window = samples[start_idx:end_idx + 1]
        present_window = [s for s in window if s.present]
        if not present_window:
            continue
        peak = max(present_window, key=lambda s: s.area_cm2)  # peak-loading instant
        cx = float(np.mean([s.centroid_cm[0] for s in present_window]))
        cy = float(np.mean([s.centroid_cm[1] for s in present_window]))
        stance_runs.append(StanceRun(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time_s=samples[start_idx].time_s,
            end_time_s=samples[end_idx].time_s,
            peak_area_cm2=peak.area_cm2,
            peak_time_s=peak.time_s,
            centroid_cm=(cx, cy),
            peak_length_cm=peak.length_cm,
            peak_width_cm=peak.width_cm,
            paw_placement_angle_deg=peak.paw_angle_deg,
        ))
    return stance_runs


def find_all_stance_runs(tracks: dict[str, list[PawFrameSample]]) -> dict[str, list[StanceRun]]:
    return {label: find_stance_runs(tracks[label]) for label in LABELS}


# ---------------------------------------------------------------------------
# Stride metrics
# ---------------------------------------------------------------------------

@dataclass
class StrideRecord:
    onset_time_s: float
    stride_time_s: float
    stride_length_cm: float
    stance_duration_s: float
    swing_duration_s: float


def compute_stride_records(
    stance_runs: list[StanceRun], belt_speed_cms: float
) -> list[StrideRecord]:
    """
    One record per completed stride: stride time = time between successive
    stance onsets; stride length = belt speed x stride time (per spec).
    """
    records = []
    for i in range(len(stance_runs) - 1):
        this_run, next_run = stance_runs[i], stance_runs[i + 1]
        stride_time = next_run.start_time_s - this_run.start_time_s
        stance_duration = this_run.end_time_s - this_run.start_time_s
        swing_duration = next_run.start_time_s - this_run.end_time_s
        records.append(StrideRecord(
            onset_time_s=this_run.start_time_s,
            stride_time_s=stride_time,
            stride_length_cm=belt_speed_cms * stride_time,
            stance_duration_s=stance_duration,
            swing_duration_s=swing_duration,
        ))
    return records


def compute_all_stride_records(
    stance_runs_by_paw: dict[str, list[StanceRun]], meta: VideoMeta
) -> dict[str, list[StrideRecord]]:
    return {
        label: compute_stride_records(runs, meta.belt_speed_cms)
        for label, runs in stance_runs_by_paw.items()
    }


# ---------------------------------------------------------------------------
# Ensemble-averaged stride cycle ("Ensemble Paws")
# ---------------------------------------------------------------------------

def compute_ensemble_cycle(
    samples: list[PawFrameSample],
    stance_runs: list[StanceRun],
    n_points: int = 101,
) -> dict:
    """
    For a single paw: take every stride window (stance onset[i] to onset
    [i+1]), resample its area-vs-time curve onto a common 0-100% stride
    axis, then average across strides. Mirrors DigiGait's 'Ensemble Paws'
    plot, which overlays N normalized stride cycles and their mean.
    """
    t_all = np.array([s.time_s for s in samples])
    a_all = np.array([s.area_cm2 if s.present else 0.0 for s in samples])
    pct_axis = np.linspace(0, 100, n_points)

    cycles = []
    for i in range(len(stance_runs) - 1):
        t0 = stance_runs[i].start_time_s
        t1 = stance_runs[i + 1].start_time_s
        if t1 <= t0:
            continue
        mask = (t_all >= t0) & (t_all <= t1)
        if mask.sum() < 2:
            continue
        t_window = t_all[mask]
        a_window = a_all[mask]
        pct_window = (t_window - t0) / (t1 - t0) * 100.0
        resampled = np.interp(pct_axis, pct_window, a_window)
        cycles.append(resampled)

    if not cycles:
        return {
            "pct_axis": pct_axis,
            "mean_curve": np.zeros(n_points),
            "std_curve": np.zeros(n_points),
            "n_strides": 0,
            "cycles": [],
        }

    cycles_arr = np.vstack(cycles)
    return {
        "pct_axis": pct_axis,
        "mean_curve": cycles_arr.mean(axis=0),
        "std_curve": cycles_arr.std(axis=0),
        "n_strides": len(cycles),
        "cycles": cycles,
    }


def compute_all_ensemble_cycles(
    tracks: dict[str, list[PawFrameSample]],
    stance_runs_by_paw: dict[str, list[StanceRun]],
) -> dict[str, dict]:
    return {
        label: compute_ensemble_cycle(tracks[label], stance_runs_by_paw[label])
        for label in LABELS
    }


# ---------------------------------------------------------------------------
# Stance width (LF-RF, LH-RH)
# ---------------------------------------------------------------------------

def _lr_axis_index(orientation: Orientation) -> int:
    """Index into a (x, y) cm tuple that corresponds to the left-right axis."""
    return 1 if orientation.axis == "horizontal" else 0


def compute_stance_width(
    tracks: dict[str, list[PawFrameSample]], orientation: Orientation
) -> pd.DataFrame:
    """
    Perpendicular-to-travel distance between LF-RF and LH-RH, for every
    frame where both paws in the pair are in contact simultaneously.
    """
    lr_idx = _lr_axis_index(orientation)
    n = len(tracks[LABELS[0]])
    rows = []
    for i in range(n):
        lf, rf = tracks["LF"][i], tracks["RF"][i]
        lh, rh = tracks["LH"][i], tracks["RH"][i]
        row = {"time_s": lf.time_s, "fore_width_cm": np.nan, "hind_width_cm": np.nan}
        if lf.present and rf.present:
            row["fore_width_cm"] = abs(lf.centroid_cm[lr_idx] - rf.centroid_cm[lr_idx])
        if lh.present and rh.present:
            row["hind_width_cm"] = abs(lh.centroid_cm[lr_idx] - rh.centroid_cm[lr_idx])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step angle (turning angle of successive footfall placements, per paw)
# ---------------------------------------------------------------------------

def compute_step_angles(stance_runs: list[StanceRun]) -> list[float]:
    """
    Angle (deg) at each interior footfall between the vector arriving from
    the previous placement and the vector leaving to the next placement.
    Straight-line stepping -> ~180 deg; sharp lateral correction -> smaller.
    """
    angles = []
    positions = [run.centroid_cm for run in stance_runs]
    for i in range(1, len(positions) - 1):
        p_prev = np.array(positions[i - 1])
        p_curr = np.array(positions[i])
        p_next = np.array(positions[i + 1])
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos_theta))))
    return angles


def compute_all_step_angles(stance_runs_by_paw: dict[str, list[StanceRun]]) -> dict[str, list[float]]:
    return {label: compute_step_angles(runs) for label, runs in stance_runs_by_paw.items()}


# ---------------------------------------------------------------------------
# Step sequence regularity
# ---------------------------------------------------------------------------

def _global_footfall_sequence(stance_runs_by_paw: dict[str, list[StanceRun]]) -> list[str]:
    events = []
    for label, runs in stance_runs_by_paw.items():
        for run in runs:
            events.append((run.start_time_s, label))
    events.sort(key=lambda e: e[0])
    return [label for _, label in events]


def compute_step_sequence_regularity(stance_runs_by_paw: dict[str, list[StanceRun]]) -> float:
    """
    % of footfalls consistent with a normal alternating quadrupedal gait.
    Method: build the global time-ordered footfall sequence across all 4
    paws, compare it against every 4-paw cyclic ordering (8 candidates:
    4 rotations x 2 directions), and score against whichever candidate
    fits best. This is a standard regularity-index approximation -- not
    DigiGait's exact proprietary formula -- so validate against the
    reference output.
    """
    sequence = _global_footfall_sequence(stance_runs_by_paw)
    if len(sequence) < 2:
        return 0.0

    candidates = []
    base = LABELS
    for start in range(4):
        rotated = base[start:] + base[:start]
        candidates.append(rotated)
        candidates.append(list(reversed(rotated)))

    best_score = 0
    for pattern in candidates:
        expected = [pattern[i % 4] for i in range(len(sequence))]
        score = sum(1 for a, b in zip(sequence, expected) if a == b)
        best_score = max(best_score, score)

    return 100.0 * best_score / len(sequence)


# ---------------------------------------------------------------------------
# Animal (body) length / width -- proxy from paw spread, see module docstring
# ---------------------------------------------------------------------------

def compute_animal_dimensions(
    tracks: dict[str, list[PawFrameSample]], orientation: Orientation
) -> dict[str, float]:
    fore_idx = 0 if orientation.axis == "vertical" else 1
    lr_idx = _lr_axis_index(orientation)

    n = len(tracks[LABELS[0]])
    lengths, widths = [], []
    for i in range(n):
        lf, rf = tracks["LF"][i], tracks["RF"][i]
        lh, rh = tracks["LH"][i], tracks["RH"][i]

        fore_present = [p for p in (lf, rf) if p.present]
        hind_present = [p for p in (lh, rh) if p.present]
        if fore_present and hind_present:
            fore_pos = np.mean([p.centroid_cm[fore_idx] for p in fore_present])
            hind_pos = np.mean([p.centroid_cm[fore_idx] for p in hind_present])
            lengths.append(abs(fore_pos - hind_pos))

        if lf.present and rf.present:
            widths.append(abs(lf.centroid_cm[lr_idx] - rf.centroid_cm[lr_idx]))
        if lh.present and rh.present:
            widths.append(abs(lh.centroid_cm[lr_idx] - rh.centroid_cm[lr_idx]))

    return {
        "animal_length_cm": float(np.mean(lengths)) if lengths else float("nan"),
        "animal_width_cm": float(np.mean(widths)) if widths else float("nan"),
    }


# ---------------------------------------------------------------------------
# Top-level aggregation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    tracks: dict[str, list[PawFrameSample]],
    meta: VideoMeta,
    orientation: Orientation,
) -> dict:
    """Run every metric and return one nested dict -- the single source
    plotting.py and the CSV export both read from."""
    dadt = compute_dadt(tracks)
    dadt_summary = dadt_min_max(dadt)
    stance_runs_by_paw = find_all_stance_runs(tracks)
    stride_records_by_paw = compute_all_stride_records(stance_runs_by_paw, meta)
    ensemble_by_paw = compute_all_ensemble_cycles(tracks, stance_runs_by_paw)
    stance_width_df = compute_stance_width(tracks, orientation)
    step_angles_by_paw = compute_all_step_angles(stance_runs_by_paw)
    regularity_pct = compute_step_sequence_regularity(stance_runs_by_paw)
    animal_dims = compute_animal_dimensions(tracks, orientation)

    return {
        "area_time_series": area_time_series_dataframe(tracks),
        "dadt": dadt,
        "dadt_summary": dadt_summary,
        "stance_runs_by_paw": stance_runs_by_paw,
        "stride_records_by_paw": stride_records_by_paw,
        "ensemble_by_paw": ensemble_by_paw,
        "stance_width_df": stance_width_df,
        "step_angles_by_paw": step_angles_by_paw,
        "regularity_pct": regularity_pct,
        "animal_dims": animal_dims,
    }


def build_summary_dataframe(all_metrics: dict, meta: VideoMeta) -> pd.DataFrame:
    """One row per paw of scalar summary metrics -- the main CSV export."""
    rows = []
    for label in LABELS:
        strides = all_metrics["stride_records_by_paw"][label]
        stance_runs = all_metrics["stance_runs_by_paw"][label]
        step_angles = all_metrics["step_angles_by_paw"][label]
        dadt_min, dadt_max = all_metrics["dadt_summary"][label]
        ensemble = all_metrics["ensemble_by_paw"][label]

        stance_col = "fore_width_cm" if label in ("LF", "RF") else "hind_width_cm"
        stance_width_mean = all_metrics["stance_width_df"][stance_col].mean()

        rows.append({
            "paw": label,
            "video": meta.path.name,
            "belt_speed_cms": meta.belt_speed_cms,
            "fps": meta.fps,
            "n_stance_phases": len(stance_runs),
            "n_strides": len(strides),
            "mean_stride_time_s": np.mean([s.stride_time_s for s in strides]) if strides else np.nan,
            "mean_stride_length_cm": np.mean([s.stride_length_cm for s in strides]) if strides else np.nan,
            "mean_stance_duration_s": np.mean([s.stance_duration_s for s in strides]) if strides else np.nan,
            "mean_swing_duration_s": np.mean([s.swing_duration_s for s in strides]) if strides else np.nan,
            # All three of the following are taken at each stance phase's
            # peak-loading (max ellipse area) frame, then averaged across
            # stance phases -- per the client's paw model, these
            # measurements are only meaningful at that specific instant.
            "peak_paw_area_cm2": np.mean([r.peak_area_cm2 for r in stance_runs]) if stance_runs else np.nan,
            "paw_length_at_peak_cm": np.mean([r.peak_length_cm for r in stance_runs]) if stance_runs else np.nan,
            "paw_width_at_peak_cm": np.mean([r.peak_width_cm for r in stance_runs]) if stance_runs else np.nan,
            "paw_placement_angle_deg": np.mean([r.paw_placement_angle_deg for r in stance_runs]) if stance_runs else np.nan,
            "mean_step_angle_deg": np.mean(step_angles) if step_angles else np.nan,
            "stance_width_cm": stance_width_mean,
            "dadt_min_cm2_s": dadt_min,
            "dadt_max_cm2_s": dadt_max,
            "ensemble_n_strides": ensemble["n_strides"],
            "step_sequence_regularity_pct": all_metrics["regularity_pct"],
            "animal_length_cm": all_metrics["animal_dims"]["animal_length_cm"],
            "animal_width_cm": all_metrics["animal_dims"]["animal_width_cm"],
        })
    return pd.DataFrame(rows)
