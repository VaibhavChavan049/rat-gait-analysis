"""
Video I/O helpers.

Responsible for two things the pipeline must never hardcode:
  1. Belt speed (cm/s)  -- parsed from the filename, with manual override.
  2. Frame rate (fps)   -- read dynamically from each video's own metadata.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2

import config


# ---------------------------------------------------------------------------
# Belt speed parsing
# ---------------------------------------------------------------------------

# Matches patterns like "24cms", "24_cms", "24cm-s", "24 cms" (case-insensitive)
# and captures the leading number (int or float).
_BELT_SPEED_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[_\-]?\s*cm\s*/?\s*s", re.IGNORECASE)


def parse_belt_speed_from_filename(filename: str) -> float | None:
    """
    Extract belt speed in cm/s from a video filename.

    Example: "a1_day11_24cms.mp4" -> 24.0

    Returns None if no match is found (caller should fall back to
    config.BELT_SPEED_OVERRIDE_CMS or prompt the user).
    """
    match = _BELT_SPEED_PATTERN.search(filename)
    if match:
        return float(match.group(1))
    return None


def resolve_belt_speed(video_path: Path, interactive: bool = True, override: float | None = None) -> float:
    """
    Resolve the belt speed for a given video using, in priority order:
      1. Filename parsing
      2. `override` (e.g. a value supplied through the web upload form)
      3. config.BELT_SPEED_OVERRIDE_CMS
      4. Interactive manual prompt (if interactive=True)

    Raises ValueError if no speed could be resolved.
    """
    speed = parse_belt_speed_from_filename(video_path.name)
    if speed is not None:
        if config.VERBOSE:
            print(f"[video_io] Belt speed parsed from filename: {speed} cm/s")
        return speed

    if override is not None:
        if config.VERBOSE:
            print(f"[video_io] Using caller-supplied belt speed override: {override} cm/s")
        return float(override)

    if config.BELT_SPEED_OVERRIDE_CMS is not None:
        if config.VERBOSE:
            print(f"[video_io] Using config.BELT_SPEED_OVERRIDE_CMS: "
                  f"{config.BELT_SPEED_OVERRIDE_CMS} cm/s")
        return float(config.BELT_SPEED_OVERRIDE_CMS)

    if interactive:
        while True:
            raw = input(
                f"Could not parse belt speed from '{video_path.name}'. "
                f"Enter belt speed in cm/s: "
            ).strip()
            try:
                return float(raw)
            except ValueError:
                print("Please enter a numeric value, e.g. 24 or 24.5")

    raise ValueError(
        f"Unable to resolve belt speed for {video_path.name}: not in filename, "
        f"no config override, and interactive=False."
    )


# ---------------------------------------------------------------------------
# Frame rate / basic video metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    belt_speed_cms: float


def read_video_metadata(
    video_path: Path, interactive: bool = True, belt_speed_override: float | None = None
) -> VideoMeta:
    """
    Open the video just long enough to read its real metadata (fps, size,
    frame count) directly from the container -- never assume a fixed fps.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not fps or fps <= 0:
        raise IOError(
            f"Video {video_path.name} reports an invalid fps ({fps}). "
            f"The container metadata may be corrupt; re-export the video."
        )

    belt_speed = resolve_belt_speed(video_path, interactive=interactive, override=belt_speed_override)

    if config.VERBOSE:
        print(f"[video_io] {video_path.name}: fps={fps:.3f}, frames={frame_count}, "
              f"size={width}x{height}, belt_speed={belt_speed} cm/s")

    return VideoMeta(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        belt_speed_cms=belt_speed,
    )


def iter_frames(video_path: Path):
    """Yield (frame_index, frame_bgr) for every frame in the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_first_frame(video_path: Path):
    """Convenience helper for calibration / orientation steps that only need one frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read first frame of: {video_path}")
    return frame


def discover_videos(video_dir: Path | None = config.VIDEO_DIR) -> list[Path]:
    """Find all candidate video files in the given directory (empty list if it's None/missing)."""
    if video_dir is None or not video_dir.is_dir():
        return []
    return [
        p for p in sorted(video_dir.iterdir())
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS
    ]


def discover_all_videos() -> list[Path]:
    """Local library (config.VIDEO_DIR, if present on this machine) plus
    anything uploaded through the web interface (config.UPLOADS_DIR)."""
    return discover_videos(config.VIDEO_DIR) + discover_videos(config.UPLOADS_DIR)
