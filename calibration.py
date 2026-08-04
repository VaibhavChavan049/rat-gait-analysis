"""
Pixel-to-cm calibration.

Three paths, in order of preference:
  1. AOI dimensions file: when a video has a sibling DigiGait "_images"
     folder (see digigait_reference.py), AOI_dimensions.txt gives the
     exact width/height of the AOI in both cm and pixels -- no detection
     needed, and it's DigiGait's own number. NOTE: on the real sample
     data this comes out anisotropic (cm/px differs between x and y by
     ~35%), almost certainly non-square pixels from the original analog
     capture -- so Calibration keeps separate x/y ratios rather than
     assuming a single scalar.
  2. Auto-detect: threshold for near-black pixels, find the tape strip
     bordering the AOI (config.CALIBRATION_TAPE_WIDTH_CM wide), measure
     its thickness in pixels.
  3. Manual AOI: user drags a rectangle over a known-width strip of tape
     on a sample frame; the shorter rectangle side is used as the pixel
     width. Fallback when auto-detection is unreliable (e.g. shadows,
     low contrast, worn tape).
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Calibration:
    cm_per_pixel_x: float
    cm_per_pixel_y: float
    tape_width_px: float | None
    method: str  # "auto" | "manual" | "aoi_dimensions_file"

    @property
    def is_isotropic(self) -> bool:
        if self.cm_per_pixel_x == 0:
            return True
        return abs(self.cm_per_pixel_y / self.cm_per_pixel_x - 1.0) < 0.01

    @property
    def cm_per_pixel(self) -> float:
        """Geometric mean -- convenience for callers that only need a
        single rough scalar (e.g. distance-threshold comparisons)."""
        return (self.cm_per_pixel_x * self.cm_per_pixel_y) ** 0.5

    def x_to_cm(self, px: float) -> float:
        return px * self.cm_per_pixel_x

    def y_to_cm(self, px: float) -> float:
        return px * self.cm_per_pixel_y

    def px_to_cm(self, px: float) -> float:
        """Isotropic convenience alias -- only exact when is_isotropic."""
        return px * self.cm_per_pixel

    def px_area_to_cm2(self, px_area: float) -> float:
        # Area under an (x, y) scale transform is exactly px_area * sx * sy
        # regardless of the shape's orientation -- true even when
        # cm_per_pixel_x != cm_per_pixel_y, so this is always exact.
        return px_area * self.cm_per_pixel_x * self.cm_per_pixel_y

    @classmethod
    def from_aoi_dimensions(
        cls, width_cm: float, height_cm: float, width_px: float, height_px: float
    ) -> "Calibration":
        return cls(
            cm_per_pixel_x=width_cm / width_px,
            cm_per_pixel_y=height_cm / height_px,
            tape_width_px=None,
            method="aoi_dimensions_file",
        )


def _tape_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(config.BLACK_TAPE_HSV_LOWER, dtype=np.uint8),
        np.array(config.BLACK_TAPE_HSV_UPPER, dtype=np.uint8),
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _detect_as_strips(mask: np.ndarray) -> float | None:
    """
    Case 1: the tape shows up as separate elongated strip contours (e.g.
    only some sides visible, or corners not touching). Width = the short
    side of each strip's oriented bounding box.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strip_widths = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if w == 0 or h == 0:
            continue
        long_side, short_side = max(w, h), min(w, h)
        if short_side == 0:
            continue
        if long_side / short_side >= 4:  # looks like a strip, not a blob
            strip_widths.append(short_side)

    if not strip_widths:
        return None
    return float(np.median(strip_widths))


def _detect_as_hollow_rectangle(mask: np.ndarray) -> float | None:
    """
    Case 2 (the common one): the tape forms a continuous rectangular
    border/frame around the AOI -- a "box" of tape, per the client's
    description. That's a single connected black region with a hole in
    the middle (the AOI). Find the outer boundary and its inner hole via
    contour hierarchy, and take the border thickness as half the gap
    between the outer and inner bounding boxes (averaged over both axes).
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

    thicknesses = []
    for i, (next_i, prev_i, first_child, parent) in enumerate(hierarchy):
        if parent != -1 or first_child == -1:
            continue  # only want top-level contours that have a hole
        outer_area = cv2.contourArea(contours[i])
        if outer_area < 500:
            continue

        # There may be several sibling holes (corners can fragment the
        # inner boundary); use the largest hole as the AOI opening.
        child = first_child
        best_child, best_child_area = None, 0
        while child != -1:
            child_area = cv2.contourArea(contours[child])
            if child_area > best_child_area:
                best_child, best_child_area = child, child_area
            child = hierarchy[child][0]  # next sibling
        if best_child is None or best_child_area < 500:
            continue

        ox, oy, ow, oh = cv2.boundingRect(contours[i])
        ix, iy, iw, ih = cv2.boundingRect(contours[best_child])
        thickness_x = (ow - iw) / 2.0
        thickness_y = (oh - ih) / 2.0
        if thickness_x > 0 and thickness_y > 0:
            thicknesses.append((thickness_x + thickness_y) / 2.0)

    if not thicknesses:
        return None
    return float(np.median(thicknesses))


def _auto_detect_tape_width_px(frame_bgr: np.ndarray) -> float | None:
    """
    Try to find the black calibration tape automatically and measure its
    width in pixels. Tries strip-shaped tape first, then falls back to a
    continuous hollow-rectangle border (the more typical "box of tape
    around the AOI" case).
    """
    mask = _tape_mask(frame_bgr)

    width_px = _detect_as_strips(mask)
    if width_px is not None:
        return width_px

    return _detect_as_hollow_rectangle(mask)


def auto_calibrate(frame_bgr: np.ndarray) -> Calibration | None:
    """Attempt automatic calibration from the black tape. Returns None on failure."""
    tape_width_px = _auto_detect_tape_width_px(frame_bgr)
    if tape_width_px is None or tape_width_px <= 0:
        return None

    cm_per_pixel = config.CALIBRATION_TAPE_WIDTH_CM / tape_width_px
    if config.VERBOSE:
        print(f"[calibration] Auto-detected tape width: {tape_width_px:.2f}px "
              f"-> {cm_per_pixel:.5f} cm/px")
    return Calibration(cm_per_pixel_x=cm_per_pixel, cm_per_pixel_y=cm_per_pixel,
                        tape_width_px=tape_width_px, method="auto")


def manual_calibrate(frame_bgr: np.ndarray) -> Calibration:
    """
    Let the user drag a rectangle over a strip of the known-width tape.
    The shorter side of the drawn rectangle is treated as the tape width
    in pixels.
    """
    window_name = "Calibration: drag a box across the tape width, then press ENTER"
    roi = cv2.selectROI(window_name, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise ValueError("Manual calibration ROI has zero size; selection was cancelled.")

    tape_width_px = float(min(w, h))
    cm_per_pixel = config.CALIBRATION_TAPE_WIDTH_CM / tape_width_px

    if config.VERBOSE:
        print(f"[calibration] Manual ROI tape width: {tape_width_px:.2f}px "
              f"-> {cm_per_pixel:.5f} cm/px")

    return Calibration(cm_per_pixel_x=cm_per_pixel, cm_per_pixel_y=cm_per_pixel,
                        tape_width_px=tape_width_px, method="manual")


def calibrate(frame_bgr: np.ndarray, allow_interactive_fallback: bool = True) -> Calibration:
    """
    Full calibration flow: try auto-detection first, fall back to manual
    AOI selection if it fails and fallback is enabled/allowed.
    """
    cal = auto_calibrate(frame_bgr)
    if cal is not None:
        return cal

    if config.VERBOSE:
        print("[calibration] Auto-detection failed.")

    if config.CALIBRATION_ALLOW_MANUAL_FALLBACK and allow_interactive_fallback:
        if config.VERBOSE:
            print("[calibration] Falling back to manual AOI selection.")
        return manual_calibrate(frame_bgr)

    raise RuntimeError(
        "Automatic tape calibration failed and manual fallback is disabled. "
        "Enable config.CALIBRATION_ALLOW_MANUAL_FALLBACK or improve lighting/contrast."
    )
