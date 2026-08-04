"""
Rat orientation (nose direction) input.

DigiGait's GUI asks the operator to click the nose position on a sample
frame so the software knows which way the rat is facing, and therefore
which quadrant of the frame corresponds to Fore vs Hind and Left vs
Right. We replicate that here.

Two supported methods (config.ORIENTATION_METHOD):
  "manual_click": show a sample frame, user clicks the nose position.
                   The direction is inferred as "the side of the frame
                   center the click is on" (up/down/left/right), which
                   matches how a single-point nose click is used in the
                   legacy workflow.
  "config":        skip the interactive step, use config.ORIENTATION_CONFIG.
"""

from dataclasses import dataclass

import cv2

import config


@dataclass
class Orientation:
    nose_direction: str  # "up" | "down" | "left" | "right"
    nose_point_px: tuple[float, float] | None = None  # (x, y) if from a click

    def fore_is_top(self) -> bool:
        """True if 'fore' (nose end) is the smaller-y (top) half of the frame."""
        return self.nose_direction == "up"

    def fore_is_left(self) -> bool:
        """True if 'fore' (nose end) is the smaller-x (left) half of the frame."""
        return self.nose_direction == "left"

    @property
    def axis(self) -> str:
        """Which image axis separates fore from hind: 'vertical' or 'horizontal'."""
        return "vertical" if self.nose_direction in ("up", "down") else "horizontal"


_VALID_DIRECTIONS = {"up", "down", "left", "right"}


def direction_from_point(x: float, y: float, frame_w: int, frame_h: int) -> str:
    """
    Convert a single nose position (from a click, or from a ground-truth
    snout coordinate -- see digigait_reference.py) into a cardinal
    direction relative to the frame center. The rat is assumed to lie
    roughly along one axis of the frame (treadmill runs lengthwise), so
    we compare how far off-center the point is along each axis and pick
    the dominant one.
    """
    cx, cy = frame_w / 2.0, frame_h / 2.0
    dx, dy = x - cx, y - cy

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def get_orientation_manual_click(frame_bgr) -> Orientation:
    """
    Show the frame, let the user click on the nose, return the inferred
    orientation. Press any key after clicking to confirm.
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    clicked = {}

    window_name = "Click the rat's NOSE position, then press any key"

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["x"], clicked["y"] = x, y

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    display = frame_bgr.copy()
    cv2.imshow(window_name, display)

    while "x" not in clicked:
        key = cv2.waitKey(20)
        if key != -1:
            break

    if "x" in clicked:
        cv2.circle(display, (clicked["x"], clicked["y"]), 6, (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        cv2.waitKey(500)

    cv2.destroyWindow(window_name)

    if "x" not in clicked:
        raise RuntimeError("No nose click was registered; cannot determine orientation.")

    direction = direction_from_point(clicked["x"], clicked["y"], frame_w, frame_h)
    if config.VERBOSE:
        print(f"[orientation] Nose clicked at ({clicked['x']}, {clicked['y']}) "
              f"-> nose_direction='{direction}'")

    return Orientation(nose_direction=direction, nose_point_px=(clicked["x"], clicked["y"]))


def get_orientation_from_config() -> Orientation:
    direction = config.ORIENTATION_CONFIG.get("nose_direction", "up")
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"config.ORIENTATION_CONFIG['nose_direction']={direction!r} is invalid; "
            f"must be one of {_VALID_DIRECTIONS}."
        )
    return Orientation(nose_direction=direction)


def resolve_orientation(frame_bgr=None) -> Orientation:
    """Dispatch to the configured orientation method."""
    method = config.ORIENTATION_METHOD
    if method == "manual_click":
        if frame_bgr is None:
            raise ValueError("manual_click orientation requires a sample frame.")
        return get_orientation_manual_click(frame_bgr)
    elif method == "config":
        return get_orientation_from_config()
    else:
        raise ValueError(f"Unknown config.ORIENTATION_METHOD: {method!r}")
