"""
Interactive helper to tune the red paw-detection HSV thresholds (and,
optionally, the black tape calibration thresholds) on a real sample
frame before running the full pipeline.

Usage:
    python tools/mask_tuner.py --video ../videos/a1_day11_24cms.mp4
    python tools/mask_tuner.py --video ../videos/a1_day11_24cms.mp4 --frame 50
    python tools/mask_tuner.py --video ../videos/a1_day11_24cms.mp4 --target tape

Drag the trackbars until the mask window cleanly isolates the red paw
regions (or the black tape). Press 'p' to print the current values in
the exact tuple format expected by config.py, or 'q'/ESC to quit.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from video_io import iter_frames  # noqa: E402


def _nothing(_):
    pass


def _make_trackbars(window, target: str):
    if target == "red":
        defaults = {
            "H1_lo": config.RED_HSV_LOWER_1[0], "H1_hi": config.RED_HSV_UPPER_1[0],
            "H2_lo": config.RED_HSV_LOWER_2[0], "H2_hi": config.RED_HSV_UPPER_2[0],
            "S_lo": config.RED_HSV_LOWER_1[1], "S_hi": config.RED_HSV_UPPER_1[1],
            "V_lo": config.RED_HSV_LOWER_1[2], "V_hi": config.RED_HSV_UPPER_1[2],
        }
    else:  # tape (black)
        defaults = {
            "H1_lo": config.BLACK_TAPE_HSV_LOWER[0], "H1_hi": config.BLACK_TAPE_HSV_UPPER[0],
            "H2_lo": 0, "H2_hi": 0,
            "S_lo": config.BLACK_TAPE_HSV_LOWER[1], "S_hi": config.BLACK_TAPE_HSV_UPPER[1],
            "V_lo": config.BLACK_TAPE_HSV_LOWER[2], "V_hi": config.BLACK_TAPE_HSV_UPPER[2],
        }

    for name, val in defaults.items():
        maxval = 180 if name.startswith("H") else 255
        cv2.createTrackbar(name, window, int(val), maxval, _nothing)


def _read_trackbars(window):
    names = ["H1_lo", "H1_hi", "H2_lo", "H2_hi", "S_lo", "S_hi", "V_lo", "V_hi"]
    return {n: cv2.getTrackbarPos(n, window) for n in names}


def _build_mask(hsv, vals, target: str):
    lower1 = np.array([vals["H1_lo"], vals["S_lo"], vals["V_lo"]], dtype=np.uint8)
    upper1 = np.array([vals["H1_hi"], vals["S_hi"], vals["V_hi"]], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1)

    if target == "red":
        lower2 = np.array([vals["H2_lo"], vals["S_lo"], vals["V_lo"]], dtype=np.uint8)
        upper2 = np.array([vals["H2_hi"], vals["S_hi"], vals["V_hi"]], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2)

    k = config.MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Interactive HSV threshold tuner")
    parser.add_argument("--video", required=True, help="Path to a sample video")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to tune on")
    parser.add_argument("--target", choices=["red", "tape"], default="red",
                         help="Tune red paw thresholds or black tape thresholds")
    args = parser.parse_args()

    video_path = Path(args.video)
    frame = None
    for idx, f in iter_frames(video_path):
        if idx == args.frame:
            frame = f
            break
    if frame is None:
        raise SystemExit(f"Could not read frame {args.frame} from {video_path}")

    window = f"Mask Tuner ({args.target}) -- press 'p' to print values, 'q' to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    _make_trackbars(window, args.target)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    while True:
        vals = _read_trackbars(window)
        mask = _build_mask(hsv, vals, args.target)
        overlay = cv2.bitwise_and(frame, frame, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack([frame, overlay, mask_bgr])
        cv2.imshow(window, stacked)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            if args.target == "red":
                print(
                    f"RED_HSV_LOWER_1 = ({vals['H1_lo']}, {vals['S_lo']}, {vals['V_lo']})\n"
                    f"RED_HSV_UPPER_1 = ({vals['H1_hi']}, {vals['S_hi']}, {vals['V_hi']})\n"
                    f"RED_HSV_LOWER_2 = ({vals['H2_lo']}, {vals['S_lo']}, {vals['V_lo']})\n"
                    f"RED_HSV_UPPER_2 = ({vals['H2_hi']}, {vals['S_hi']}, {vals['V_hi']})"
                )
            else:
                print(
                    f"BLACK_TAPE_HSV_LOWER = ({vals['H1_lo']}, {vals['S_lo']}, {vals['V_lo']})\n"
                    f"BLACK_TAPE_HSV_UPPER = ({vals['H1_hi']}, {vals['S_hi']}, {vals['V_hi']})"
                )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
