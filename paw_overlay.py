"""
Visual QC helper: draws the detected/labeled paw ellipses directly on
top of real video frames, so you can SEE what the pipeline is tracking
frame by frame -- not just read numbers off a summary plot.

Produces two things per video:
  - a short annotated MP4 clip: each paw's fitted ellipse + LF/RF/LH/RH
    label drawn on every frame, upscaled so it's readable -- this IS the
    paw movement, visualized directly.
  - one annotated snapshot JPEG (first frame where all 4 paws are
    detected) for a quick still-photo look.
"""

import cv2
import numpy as np
from PIL import Image

from orientation import Orientation
from paw_detection import detect_paw_blobs
from paw_labeling import BodyCenterTracker, label_blobs_in_frame
from video_io import iter_frames

# BGR (OpenCV order), matches config.DIGIGAIT_LINE_COLORS' intent.
_BGR_COLORS = {
    "LF": (0, 255, 255),    # yellow
    "RF": (255, 255, 0),    # cyan
    "LH": (255, 0, 255),    # magenta
    "RH": (255, 255, 255),  # white
}


def _draw_labeled_blobs(frame_bgr: np.ndarray, labeled_blobs: dict) -> np.ndarray:
    annotated = frame_bgr.copy()
    for label, blob in labeled_blobs.items():
        color = _BGR_COLORS[label]
        if blob.ellipse is not None:
            cv2.ellipse(annotated, blob.ellipse, color, 2)
        else:
            box = cv2.boxPoints(blob.min_area_rect).astype(np.int32)
            cv2.drawContours(annotated, [box], 0, color, 2)
        cx, cy = int(blob.centroid_px[0]), int(blob.centroid_px[1])
        cv2.putText(annotated, label, (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return annotated


def save_snapshot(frame_bgr, output_path, scale: int = 4):
    """Write an already-annotated frame (see paw_labeling.build_paw_tracks_and_visuals) to disk as a JPEG."""
    if frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_path), resized)
    return output_path


def save_clip(frames_rgb: list, output_path, frame_duration_ms: int = 40):
    """Write already-annotated RGB frames (see paw_labeling.build_paw_tracks_and_visuals) to disk as a GIF."""
    if not frames_rgb:
        return None
    pil_frames = [Image.fromarray(f) for f in frames_rgb]
    pil_frames[0].save(
        output_path, format="GIF", save_all=True, append_images=pil_frames[1:],
        duration=frame_duration_ms, loop=0,
    )
    return output_path


def generate_annotated_clip(video_path, orientation: Orientation, output_path,
                             max_frames: int = 150, scale: int = 3, frame_duration_ms: int = 40):
    """
    Standalone version: decodes and re-runs detection on the video
    itself, separately from track-building. server.py does NOT use this
    -- it uses paw_labeling.build_paw_tracks_and_visuals(capture_visuals
    =True) + save_clip() so detection only runs once per video instead
    of three times (see that function's docstring for why that matters
    on a resource-constrained server). Kept here for standalone/CLI use.

    Write a short annotated GIF: fitted ellipse + label for every
    detected paw, on every frame (up to max_frames). AOI-cropped source
    frames are tiny, so `scale` upsamples the output for readability.

    GIF (not MP4) so it displays reliably everywhere -- st.image(),
    a plain double-click in Finder, a browser -- with no video-codec
    compatibility questions. Playback speed is a fixed, watchable rate
    (frame_duration_ms), not the source video's true fps (which can be
    over 100fps on this hardware -- too fast to be useful to look at).
    """
    frames_rgb = []

    body_tracker = None
    for frame_idx, frame in iter_frames(video_path):
        if len(frames_rgb) >= max_frames:
            break
        frame_h, frame_w = frame.shape[:2]
        if body_tracker is None:
            body_tracker = BodyCenterTracker((frame_w / 2.0, frame_h / 2.0))

        blobs = detect_paw_blobs(frame)
        body_center = body_tracker.update([b.centroid_px for b in blobs])
        labeled = label_blobs_in_frame(blobs, body_center, orientation, frame_w, frame_h)

        annotated = _draw_labeled_blobs(frame, labeled)
        if scale != 1:
            annotated = cv2.resize(annotated, (frame_w * scale, frame_h * scale),
                                    interpolation=cv2.INTER_NEAREST)

        frames_rgb.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    if not frames_rgb:
        return None, 0

    pil_frames = [Image.fromarray(f) for f in frames_rgb]
    pil_frames[0].save(
        output_path, format="GIF", save_all=True, append_images=pil_frames[1:],
        duration=frame_duration_ms, loop=0,
    )
    return output_path, len(pil_frames)


def generate_annotated_snapshot(video_path, orientation: Orientation, output_path,
                                 scale: int = 4, scan_limit: int = 400):
    """
    Save one annotated still frame -- the first frame (within
    scan_limit frames) where all 4 paws are detected simultaneously, or
    whichever frame has the most paws detected if all 4 never coincide.
    """
    body_tracker = None
    best_frame, best_labeled, best_count = None, None, -1

    for frame_idx, frame in iter_frames(video_path):
        if frame_idx >= scan_limit:
            break
        frame_h, frame_w = frame.shape[:2]
        if body_tracker is None:
            body_tracker = BodyCenterTracker((frame_w / 2.0, frame_h / 2.0))

        blobs = detect_paw_blobs(frame)
        body_center = body_tracker.update([b.centroid_px for b in blobs])
        labeled = label_blobs_in_frame(blobs, body_center, orientation, frame_w, frame_h)

        if len(labeled) > best_count:
            best_frame, best_labeled, best_count = frame, labeled, len(labeled)
        if len(labeled) == 4:
            break

    if best_frame is None:
        return None

    annotated = _draw_labeled_blobs(best_frame, best_labeled)
    h, w = annotated.shape[:2]
    annotated = cv2.resize(annotated, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_path), annotated)
    return output_path
