"""
Per-frame red paw-contact blob detection.

The processed video highlights paw-belt contact area in red. For each
frame we:
  1. Threshold for red in HSV (two ranges, since red wraps hue 0/180).
  2. Clean the mask with morphological close/open.
  3. Find contours, filter by minimum area.
  4. Fit a best-fit ELLIPSE to each contour (cv2.fitEllipse) -- this is
     the client's paw model, matching DigiGait: the paw is represented
     as an ellipse with a long axis and a short axis, not a raw pixel
     blob. Hind paws come out long/narrow (major >> minor); front paws
     come out closer to circular (major ~= minor).
  5. Keep up to config.MAX_PAWS_PER_FRAME largest blobs (by ellipse
     area, since that's the signal everything downstream is built on).

This module only detects blobs -- it does NOT know which blob is which
paw. That identity assignment happens in paw_labeling.py, so the two
concerns (finding red regions vs. deciding "this is LF") stay decoupled
and each can be re-tuned independently.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config

# cv2.fitEllipse requires at least 5 contour points to solve for the
# ellipse; below that we fall back to the minAreaRect box as a stand-in
# "ellipse" (a rectangle's w/h used as major/minor, with the matching
# pi/4 * w * h area formula so the two paths stay numerically consistent).
_MIN_POINTS_FOR_ELLIPSE_FIT = 5


@dataclass
class PawBlob:
    """One detected red contact region in a single frame, modeled as a best-fit ellipse."""
    centroid_px: tuple[float, float]      # (x, y) -- contour centroid, used for position/tracking
    area_px: float                        # raw contour pixel area (kept for QC/debugging only)
    contour: np.ndarray
    bbox_px: tuple[int, int, int, int]    # x, y, w, h
    min_area_rect: tuple                  # (center, (w, h), angle_deg) -- fallback / QC only
    ellipse: tuple | None                 # (center, (major, minor), angle_deg) from cv2.fitEllipse, or None if fallback was used
    ellipse_major_px: float               # long axis length (paw length)
    ellipse_minor_px: float               # short axis length (paw width)
    ellipse_major_axis_angle_deg: float   # direction of the MAJOR axis specifically (see note below)
    ellipse_area_px: float                # pi * (major/2) * (minor/2) -- THE area signal used everywhere downstream


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    k = config.MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def build_red_mask_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Bright, saturated-red mask (paint/marker style highlighting). Good
    default for videos where paw contact is drawn in with a vivid red
    overlay color. Two hue ranges since red wraps HSV hue 0/180.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array(config.RED_HSV_LOWER_1, dtype=np.uint8),
        np.array(config.RED_HSV_UPPER_1, dtype=np.uint8),
    )
    mask2 = cv2.inRange(
        hsv,
        np.array(config.RED_HSV_LOWER_2, dtype=np.uint8),
        np.array(config.RED_HSV_UPPER_2, dtype=np.uint8),
    )
    mask = cv2.bitwise_or(mask1, mask2)
    return _clean_mask(mask)


def build_red_mask_rgb_ratio(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Subtler-redness mask for real DigiGait footage: on real sample
    frames, paw-belt contact shows as a light pink/salmon discoloration
    (pressure-induced skin reddening), NOT a bright painted-on red --
    max observed R/G ratio in a real sample frame was only ~1.68, so
    plain bright-red HSV thresholding (tuned for paint/markers) misses
    it almost entirely.

    Flags a pixel as paw-contact when its red channel is elevated
    relative to green by at least config.RGB_RATIO_RG_THRESHOLD, and
    bright enough to not be background shadow noise. The threshold was
    empirically tuned against a real frame (a1_day11_24cms-100.jpg) by
    checking that the resulting blobs' positions matched DigiGait's own
    ground-truth labeled mask (bw_image_100.jpg) for that same frame --
    NOT derived from RGB_Values_for_this_mouse_paw.txt (see
    digigait_reference.py's module docstring for why that file's values
    didn't check out). Re-validate/retune per dataset with
    tools/mask_tuner.py --target rgb_ratio.
    """
    b, g, r = cv2.split(frame_bgr.astype(np.float32))
    eps = 1e-6
    rg_ratio = r / (g + eps)

    cond = (rg_ratio >= config.RGB_RATIO_RG_THRESHOLD) & (r >= config.RGB_RATIO_MIN_RED)
    mask = (cond.astype(np.uint8)) * 255
    return _clean_mask(mask)


def build_red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Dispatch to the configured detection method (config.PAW_DETECTION_METHOD)."""
    if config.PAW_DETECTION_METHOD == "rgb_ratio":
        return build_red_mask_rgb_ratio(frame_bgr)
    return build_red_mask_hsv(frame_bgr)


# cv2.fitEllipse's least-squares solve can produce a degenerate, wildly
# oversized ellipse for edge-clipped or near-collinear contours -- seen
# on real footage: a paw blob touching the AOI's left edge (bounding box
# only 13x18px) produced a "major axis" of 375px, ~17x its own bounding
# box diagonal. minAreaRect (no least-squares involved, always exactly
# the point set's minimal enclosing rotated rectangle) can't degenerate
# this way, so it's used as the sanity reference: a legitimate ellipse
# fit's major axis should never greatly exceed the bounding rect's
# diagonal.
_MAX_ELLIPSE_MAJOR_VS_BBOX_DIAGONAL = 2.0


def _fit_paw_ellipse(contour: np.ndarray, min_area_rect: tuple):
    """
    Fit a best-fit ellipse to the paw contour. Returns
    (ellipse_or_None, major_px, minor_px, major_axis_angle_deg, ellipse_area_px).

    Both cv2.fitEllipse and cv2.minAreaRect return their angle as the
    rotation of whichever axis is listed *first* in the (d1, d2) / (w, h)
    pair -- that's not necessarily the major/long axis. We resolve that
    here once, so every downstream consumer can just trust that the
    returned angle is always the major axis's direction.
    """
    (_, _), (rect_w, rect_h), _ = min_area_rect
    rect_diagonal = float(np.hypot(rect_w, rect_h))

    ellipse = None
    if len(contour) >= _MIN_POINTS_FOR_ELLIPSE_FIT:
        candidate = cv2.fitEllipse(contour)
        (_, _), (cand_d1, cand_d2), _ = candidate
        if max(cand_d1, cand_d2) <= _MAX_ELLIPSE_MAJOR_VS_BBOX_DIAGONAL * rect_diagonal:
            ellipse = candidate

    if ellipse is not None:
        (_, _), (d1, d2), raw_angle_deg = ellipse
    else:
        # Degenerate/tiny/edge-clipped contour: either too few points for
        # cv2.fitEllipse, or its fit failed the sanity check above. Use
        # the oriented bounding box as a stand-in ellipse so
        # area/length/width/angle stay defined and physically plausible.
        (_, _), (d1, d2), raw_angle_deg = min_area_rect

    if d1 >= d2:
        major_px, minor_px = d1, d2
        major_axis_angle_deg = raw_angle_deg
    else:
        major_px, minor_px = d2, d1
        major_axis_angle_deg = raw_angle_deg + 90.0

    ellipse_area_px = np.pi * (major_px / 2.0) * (minor_px / 2.0)
    return ellipse, major_px, minor_px, major_axis_angle_deg, ellipse_area_px


def detect_paw_blobs(frame_bgr: np.ndarray) -> list[PawBlob]:
    """
    Detect up to config.MAX_PAWS_PER_FRAME red paw-contact blobs in a
    single frame, each modeled as a best-fit ellipse, sorted by ellipse
    area largest-first.
    """
    mask = build_red_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_PAW_BLOB_AREA_PX:
            continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        bbox = cv2.boundingRect(c)
        min_area_rect = cv2.minAreaRect(c)
        ellipse, major_px, minor_px, major_axis_angle_deg, ellipse_area_px = _fit_paw_ellipse(c, min_area_rect)

        blobs.append(PawBlob(
            centroid_px=(cx, cy),
            area_px=area,
            contour=c,
            bbox_px=bbox,
            min_area_rect=min_area_rect,
            ellipse=ellipse,
            ellipse_major_px=major_px,
            ellipse_minor_px=minor_px,
            ellipse_major_axis_angle_deg=major_axis_angle_deg,
            ellipse_area_px=ellipse_area_px,
        ))

    blobs.sort(key=lambda b: b.ellipse_area_px, reverse=True)
    return blobs[:config.MAX_PAWS_PER_FRAME]
