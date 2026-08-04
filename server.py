"""
Flask backend for the Rat Gait Analysis desktop-style interface.

Deliberately plain Flask (no heavy framework) so this is easy to deploy
later -- `flask run`, gunicorn, a Docker container, whatever. Run
locally with:

    ./.venv/bin/python server.py

then open http://localhost:5050 in a browser.
"""

import traceback
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

import config
import digigait_reference
from calibration import Calibration, auto_calibrate
from metrics import build_summary_dataframe, compute_all_metrics
from orientation import Orientation
from paw_labeling import build_paw_tracks_and_visuals
from paw_overlay import save_clip, save_snapshot
from plotting import generate_all_plots
from video_io import discover_all_videos, parse_belt_speed_from_filename, read_first_frame, read_video_metadata

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES


def _video_path_by_name(name: str) -> Path:
    for v in discover_all_videos():
        if v.name == name:
            return v
    raise FileNotFoundError(f"No such video: {name}")


def _resolve_metadata(video_path: Path) -> digigait_reference.DigiGaitMetadata | None:
    """This video's own DigiGait output if it has one, else a shared
    default borrowed from other videos in the same session folder (see
    digigait_reference.find_session_defaults for why that's justified)."""
    own = digigait_reference.load_metadata(video_path)
    if own is not None:
        return own
    return digigait_reference.find_session_defaults(video_path.parent)


@app.route("/")
def index():
    folder_label = str(config.VIDEO_DIR) if config.VIDEO_DIR else "(none on this server -- upload videos below)"
    return render_template("index.html", folder_name=folder_label)


@app.route("/api/videos")
def api_videos():
    videos = discover_all_videos()
    out = []
    for v in videos:
        own = digigait_reference.load_metadata(v)
        meta = own if own is not None else digigait_reference.find_session_defaults(v.parent)
        out.append({
            "name": v.name,
            "uploaded": v.parent == config.UPLOADS_DIR,
            "has_own_reference": own is not None,
            "auto_calibrated": meta is not None,
            "needs_orientation": meta is None or meta.orientation is None,
            "needs_calibration": meta is None or meta.calibration is None,
        })
    return jsonify({
        "folder": str(config.VIDEO_DIR) if config.VIDEO_DIR else None,
        "videos": out,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in config.VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{suffix}'. Use .mp4, .avi, or .mov."}), 400

    dest = config.UPLOADS_DIR / filename
    stem, i = dest.stem, 1
    while dest.exists():
        dest = config.UPLOADS_DIR / f"{stem}_{i}{suffix}"
        i += 1

    file.save(dest)
    return jsonify({"name": dest.name})


@app.errorhandler(413)
def too_large(_):
    limit_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({"error": f"File too large -- limit is {limit_mb} MB."}), 413


@app.route("/api/preview/<video_name>")
def api_preview(video_name):
    """First frame of a video, as a PNG, for the orientation/calibration picker."""
    video_path = _video_path_by_name(video_name)
    frame = read_first_frame(video_path)
    out_path = config.OUTPUT_DIR / "plots" / f"_preview_{video_path.stem}.png"
    cv2.imwrite(str(out_path), frame)
    return send_from_directory(out_path.parent, out_path.name)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


@app.route("/api/run", methods=["POST"])
def api_run():
    try:
        return _run_pipeline(request.get_json(force=True))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def _run_pipeline(body: dict):
    video_name = body.get("video")
    nose_direction = body.get("nose_direction")
    belt_speed_override = body.get("belt_speed")
    tape_width_px = body.get("tape_width_px")

    video_path = _video_path_by_name(video_name)
    digigait_meta = _resolve_metadata(video_path)

    try:
        meta = read_video_metadata(
            video_path, interactive=False,
            belt_speed_override=float(belt_speed_override) if belt_speed_override else None,
        )
    except ValueError:
        return jsonify({
            "error": "belt_speed_required",
            "message": "Belt speed isn't in the filename -- enter it (cm/s) and run again.",
        }), 409
    if digigait_meta is not None and digigait_meta.belt_speed_cms is not None:
        meta.belt_speed_cms = digigait_meta.belt_speed_cms

    first_frame = read_first_frame(video_path)

    if digigait_meta is not None and digigait_meta.calibration is not None:
        calibration = digigait_meta.calibration
    else:
        # No DigiGait reference/session default -- try auto tape
        # detection, then fall back to a manually measured pixel
        # distance (two points clicked on the preview image in the UI).
        calibration = auto_calibrate(first_frame)
        if calibration is None and tape_width_px:
            cm_per_pixel = config.CALIBRATION_TAPE_WIDTH_CM / float(tape_width_px)
            calibration = Calibration(
                cm_per_pixel_x=cm_per_pixel, cm_per_pixel_y=cm_per_pixel,
                tape_width_px=float(tape_width_px), method="manual_web",
            )
        if calibration is None:
            return jsonify({
                "error": "calibration_required",
                "message": "Could not auto-detect the calibration tape in this video, and no "
                           "reference/session default is available. Click two points across the "
                           "tape width on the preview image, then run again.",
            }), 409

    if digigait_meta is not None and digigait_meta.orientation is not None:
        orientation = digigait_meta.orientation
    elif nose_direction:
        orientation = Orientation(nose_direction=nose_direction)
    else:
        return jsonify({
            "error": "orientation_required",
            "message": "No reference/session default orientation available -- pick a nose direction.",
        }), 409

    # Single pass over the video: builds the metrics tracks AND collects
    # the snapshot/clip frames at the same time (see that function's
    # docstring -- this used to be 3 separate full-video decode+detect
    # passes, which is what was pushing wall-clock time past Render's
    # free-tier request timeout on real videos).
    tracks, snapshot_frame, clip_frames_rgb = build_paw_tracks_and_visuals(
        video_path, meta, calibration, orientation, capture_visuals=True,
    )
    all_metrics = compute_all_metrics(tracks, meta, orientation)
    plot_paths = generate_all_plots(all_metrics, meta, config.PLOTS_DIR)
    summary_df = build_summary_dataframe(all_metrics, meta)

    stem = video_path.stem
    summary_csv_path = config.CSV_DIR / f"{stem}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    all_metrics["area_time_series"].to_csv(config.CSV_DIR / f"{stem}_area_time_series.csv", index=False)

    snapshot_path = config.PLOTS_DIR / f"{stem}_paw_overlay_snapshot.jpg"
    save_snapshot(snapshot_frame, snapshot_path)

    clip_path = config.PLOTS_DIR / f"{stem}_paw_overlay.gif"
    save_clip(clip_frames_rgb, clip_path)

    comparison_rows = None
    if digigait_meta is not None and not digigait_meta.is_session_default and digigait_meta.images_dir is not None:
        indices_df = digigait_reference.load_reference_indices(digigait_meta.images_dir, stem)
        if indices_df is not None:
            comparison_df = digigait_reference.compare_summary_to_reference(summary_df, indices_df)
            if not comparison_df.empty:
                comparison_rows = comparison_df.round(3).to_dict(orient="records")

    def _out_url(p: Path) -> str:
        return f"/outputs/{Path(p).relative_to(config.OUTPUT_DIR).as_posix()}"

    return jsonify({
        "video": video_name,
        "fps": round(meta.fps, 2),
        "belt_speed_cms": meta.belt_speed_cms,
        "cm_per_pixel_x": round(calibration.cm_per_pixel_x, 6),
        "cm_per_pixel_y": round(calibration.cm_per_pixel_y, 6),
        "used_reference_data": digigait_meta is not None and not digigait_meta.is_session_default,
        "used_session_default": digigait_meta is not None and digigait_meta.is_session_default,
        "snapshot_url": _out_url(snapshot_path),
        "clip_url": _out_url(clip_path),
        "gait_signals_url": _out_url(plot_paths["gait_signals"]),
        "ensemble_paws_url": _out_url(plot_paths["ensemble_paws"]),
        "posture_url": _out_url(plot_paths["posture"]),
        "summary_rows": summary_df.round(4).to_dict(orient="records"),
        "summary_csv_url": _out_url(summary_csv_path),
        "comparison_rows": comparison_rows,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
