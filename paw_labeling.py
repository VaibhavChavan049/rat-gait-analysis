"""
Assigns each per-frame red blob (from paw_detection.py) to one of
LF/RF/LH/RH, and builds continuous per-paw time series across a whole
video.

Labeling method (quadrant-based, per spec): the frame is split into a
fore/hind half and a left/right half, relative to the rat's nose
direction (orientation.py). A blob's label is whichever of the four
quadrants it's closest to. Because not every paw touches the belt in
every frame, the quadrant split is anchored to a smoothed running
estimate of the rat's body center (not a fixed point), so it keeps
working as the rat drifts fore/aft or side to side on the belt.

Camera note: the camera looks UP at the rat's underside (ventral view),
which mirrors left/right relative to a from-above (dorsal) view.
config.MIRROR_LEFT_RIGHT compensates for this -- flip it if LF/RF (or
LH/RH) come out swapped when validating against the reference DigiGait
output.

This module deliberately keeps "find the blobs" (paw_detection.py) and
"decide what to call each blob" (this file) separate, so the labeling
logic can be swapped out without touching detection.

Paw shape/area model: each paw is represented every frame as a best-fit
ellipse (see paw_detection.PawBlob), not a raw pixel count. The area
carried into PawFrameSample.area_cm2 is the ellipse's area
(pi * major/2 * minor/2) -- that's the signal Dynamic Gait Signals,
dA/dT, and stance-phase detection are all built on, per the client's
paw model. Hind paws should come out long/narrow (major >> minor); front
paws closer to circular.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from calibration import Calibration
from orientation import Orientation
from paw_detection import _MAX_ELLIPSE_MAJOR_VS_BBOX_DIAGONAL, PawBlob, detect_paw_blobs
from video_io import VideoMeta, iter_frames

LABELS = ["LF", "RF", "LH", "RH"]

# Ideal quadrant unit vectors in (fore_axis, lr_axis) coordinates, where
#   fore_axis: +1 = fore (nose end),        -1 = hind
#   lr_axis:   +1 = rat's right,             -1 = rat's left  (after mirror correction)
_IDEAL_QUADRANT = {
    "LF": (+1, -1),
    "RF": (+1, +1),
    "LH": (-1, -1),
    "RH": (-1, +1),
}

_TRAVEL_DIRECTION = {
    "up": np.array([0.0, -1.0]),
    "down": np.array([0.0, 1.0]),
    "left": np.array([-1.0, 0.0]),
    "right": np.array([1.0, 0.0]),
}


@dataclass
class PawFrameSample:
    """One paw's state at one frame (present=False if not touching down)."""
    frame_idx: int
    time_s: float
    present: bool
    centroid_cm: tuple[float, float] | None = None
    area_cm2: float = 0.0
    length_cm: float = 0.0
    width_cm: float = 0.0
    paw_angle_deg: float = 0.0


class BodyCenterTracker:
    """Smoothed running estimate of the rat's body center in pixel space."""

    def __init__(self, initial_center: tuple[float, float], alpha: float = 0.2):
        self.center = initial_center
        self.alpha = alpha

    def update(self, blob_centroids: list[tuple[float, float]]) -> tuple[float, float]:
        if blob_centroids:
            mean_x = float(np.mean([c[0] for c in blob_centroids]))
            mean_y = float(np.mean([c[1] for c in blob_centroids]))
            cx, cy = self.center
            self.center = (
                cx + self.alpha * (mean_x - cx),
                cy + self.alpha * (mean_y - cy),
            )
        return self.center


def _blob_axis_coords(
    blob: PawBlob,
    body_center: tuple[float, float],
    orientation: Orientation,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """Convert a blob centroid into normalized (fore_axis, lr_axis) coords."""
    bx, by = blob.centroid_px
    cx, cy = body_center
    dx, dy = bx - cx, by - cy

    if orientation.axis == "vertical":
        raw_fore = -dy if orientation.nose_direction == "up" else dy
        raw_lr = dx
        norm_fore = raw_fore / (frame_h / 2.0)
        norm_lr = raw_lr / (frame_w / 2.0)
    else:
        raw_fore = -dx if orientation.nose_direction == "left" else dx
        raw_lr = dy
        norm_fore = raw_fore / (frame_w / 2.0)
        norm_lr = raw_lr / (frame_h / 2.0)

    if config.MIRROR_LEFT_RIGHT:
        norm_lr = -norm_lr

    return norm_fore, norm_lr


def label_blobs_in_frame(
    blobs: list[PawBlob],
    body_center: tuple[float, float],
    orientation: Orientation,
    frame_w: int,
    frame_h: int,
) -> dict[str, PawBlob]:
    """
    Assign each of the (0-4) detected blobs to a unique label via min-cost
    matching against the four ideal quadrant corners. Returns {label: blob}
    only for labels matched this frame.
    """
    if not blobs:
        return {}

    n = len(blobs)
    cost = np.zeros((n, 4))
    for i, blob in enumerate(blobs):
        fore, lr = _blob_axis_coords(blob, body_center, orientation, frame_w, frame_h)
        for j, label in enumerate(LABELS):
            ideal_fore, ideal_lr = _IDEAL_QUADRANT[label]
            cost[i, j] = (fore - ideal_fore) ** 2 + (lr - ideal_lr) ** 2

    row_idx, col_idx = linear_sum_assignment(cost)
    return {LABELS[c]: blobs[r] for r, c in zip(row_idx, col_idx)}


_MIN_POINTS_FOR_ELLIPSE_FIT = 5


def _major_axis_vs_travel_angle(long_vec: np.ndarray, orientation: Orientation) -> float:
    travel = _TRAVEL_DIRECTION[orientation.nose_direction]
    dot = float(np.dot(long_vec, travel))
    cross = float(long_vec[0] * travel[1] - long_vec[1] * travel[0])
    angle_deg = float(np.degrees(np.arctan2(cross, dot)))
    # Fold into (-90, 90] since a paw's long axis has no inherent direction.
    while angle_deg <= -90:
        angle_deg += 180
    while angle_deg > 90:
        angle_deg -= 180
    return angle_deg


def _paw_ellipse_length_width_angle(
    blob: PawBlob, orientation: Orientation, calibration: Calibration
) -> tuple[float, float, float]:
    """
    Derive paw length (ellipse major axis), width (minor axis), and the
    major axis's rotation angle relative to a line drawn through the
    animal (approximated as the direction of travel -- the rat runs
    lengthwise along the belt, so this line and the body's longitudinal
    axis coincide). This angle, evaluated ONLY at each stance phase's
    peak-area frame, is the client's "Paw Placement Angle" -- see
    metrics.find_stance_runs, which picks that peak frame.

    Calibration can be anisotropic (cm_per_pixel_x != cm_per_pixel_y --
    seen on real AOI-dimensions-file calibration, likely non-square
    pixels from the original analog capture). Area scales simply under
    anisotropic scaling regardless of orientation, but a rotated
    ellipse's axis lengths and angle do NOT scale simply -- so in that
    case we refit the ellipse directly on the contour after scaling it
    into cm-space, rather than just multiplying the pixel-space ellipse.
    """
    if calibration.is_isotropic:
        theta = np.radians(blob.ellipse_major_axis_angle_deg)
        long_vec = np.array([np.cos(theta), np.sin(theta)])
        angle_deg = _major_axis_vs_travel_angle(long_vec, orientation)
        return (
            blob.ellipse_major_px * calibration.cm_per_pixel_x,
            blob.ellipse_minor_px * calibration.cm_per_pixel_x,
            angle_deg,
        )

    pts_cm = blob.contour.reshape(-1, 2).astype(np.float32)
    pts_cm[:, 0] *= calibration.cm_per_pixel_x
    pts_cm[:, 1] *= calibration.cm_per_pixel_y

    # minAreaRect is numerically robust (no least-squares fit, always
    # exactly the point set's minimal enclosing rotated rectangle) --
    # used both as the fallback and as a sanity reference for the
    # ellipse fit. See paw_detection._fit_paw_ellipse for why: on real
    # footage, cv2.fitEllipse produced a ~17x-oversized major axis for a
    # contour clipped at the AOI edge.
    min_area_rect_cm = cv2.minAreaRect(pts_cm)
    (_, _), (rect_w, rect_h), _ = min_area_rect_cm
    rect_diagonal = float(np.hypot(rect_w, rect_h))

    fit_ok = False
    if len(pts_cm) >= _MIN_POINTS_FOR_ELLIPSE_FIT:
        (_, _), (cand_d1, cand_d2), _ = cv2.fitEllipse(pts_cm)
        fit_ok = max(cand_d1, cand_d2) <= _MAX_ELLIPSE_MAJOR_VS_BBOX_DIAGONAL * rect_diagonal

    if fit_ok:
        (_, _), (d1, d2), raw_angle_deg = cv2.fitEllipse(pts_cm)
    else:
        (_, _), (d1, d2), raw_angle_deg = min_area_rect_cm

    if d1 >= d2:
        major_cm, minor_cm, axis_angle_deg = float(d1), float(d2), raw_angle_deg
    else:
        major_cm, minor_cm, axis_angle_deg = float(d2), float(d1), raw_angle_deg + 90.0

    theta = np.radians(axis_angle_deg)
    long_vec = np.array([np.cos(theta), np.sin(theta)])
    angle_deg = _major_axis_vs_travel_angle(long_vec, orientation)

    return major_cm, minor_cm, angle_deg


def build_paw_tracks(
    video_path,
    meta: VideoMeta,
    calibration: Calibration,
    orientation: Orientation,
) -> dict[str, list[PawFrameSample]]:
    """
    Run detection + labeling across every frame of the video, returning
    one continuous time series per paw label (LF/RF/LH/RH). Frames where
    a given paw isn't in contact with the belt get present=False samples
    with zero area, so downstream metrics can compute stance/swing phases
    and dA/dt cleanly over a uniform timeline.
    """
    tracks: dict[str, list[PawFrameSample]] = {label: [] for label in LABELS}
    body_tracker: BodyCenterTracker | None = None

    for frame_idx, frame in iter_frames(video_path):
        frame_h, frame_w = frame.shape[:2]
        time_s = frame_idx / meta.fps

        if body_tracker is None:
            body_tracker = BodyCenterTracker((frame_w / 2.0, frame_h / 2.0))

        blobs = detect_paw_blobs(frame)
        body_center = body_tracker.update([b.centroid_px for b in blobs])
        labeled = label_blobs_in_frame(blobs, body_center, orientation, frame_w, frame_h)

        for label in LABELS:
            blob = labeled.get(label)
            if blob is None:
                tracks[label].append(PawFrameSample(frame_idx=frame_idx, time_s=time_s, present=False))
                continue

            cx_cm = calibration.x_to_cm(blob.centroid_px[0])
            cy_cm = calibration.y_to_cm(blob.centroid_px[1])
            # Area is the fitted ellipse's area (pi * major/2 * minor/2),
            # per the client's paw model -- not the raw contour pixel
            # count. This is the signal every downstream metric is built on.
            area_cm2 = calibration.px_area_to_cm2(blob.ellipse_area_px)
            length_cm, width_cm, angle_deg = _paw_ellipse_length_width_angle(
                blob, orientation, calibration
            )

            tracks[label].append(PawFrameSample(
                frame_idx=frame_idx,
                time_s=time_s,
                present=True,
                centroid_cm=(cx_cm, cy_cm),
                area_cm2=area_cm2,
                length_cm=length_cm,
                width_cm=width_cm,
                paw_angle_deg=angle_deg,
            ))

    if config.VERBOSE:
        for label in LABELS:
            n_present = sum(1 for s in tracks[label] if s.present)
            print(f"[paw_labeling] {label}: {n_present}/{len(tracks[label])} frames in contact")

    return tracks
