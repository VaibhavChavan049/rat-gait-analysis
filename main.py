"""
Pipeline entry point.

Usage:
    python main.py                      # process every video in config.VIDEO_DIR
    python main.py --video path/to.mp4  # process a single video
    python main.py --no-interactive     # skip manual click/prompt steps
                                         # (requires ORIENTATION_METHOD="config"
                                         # and a resolvable belt speed per video;
                                         # calibration still falls back to manual
                                         # ROI selection unless auto-detect succeeds)

For each video this runs the full chain:
  video_io            -> read fps / belt speed
  digigait_reference  -> if a sibling "<stem>_images/" DigiGait output
                          folder exists next to the video, use ITS exact
                          calibration/belt-speed/orientation instead of
                          guessing/asking -- see digigait_reference.py
  calibration         -> (fallback) px-per-cm from the black tape
  orientation         -> (fallback) nose direction
  paw_labeling        -> per-frame red-blob detection + LF/RF/LH/RH assignment
  metrics             -> all gait/momentum metrics
  plotting            -> Dynamic Gait Signals, Ensemble Paws, Posture Plot
  CSV export          -> one summary row per paw, plus per-video time series/detail files
  validation          -> if DigiGait's own INDICES_<stem>.xls reference exists,
                          also export a side-by-side comparison CSV
"""

import argparse
from pathlib import Path

import pandas as pd

import config
import digigait_reference
from calibration import calibrate
from metrics import build_summary_dataframe, compute_all_metrics
from orientation import resolve_orientation
from paw_labeling import build_paw_tracks
from plotting import generate_all_plots
from video_io import discover_videos, read_first_frame, read_video_metadata


def process_video(video_path: Path, interactive: bool = True) -> pd.DataFrame:
    print(f"\n=== Processing {video_path.name} ===")

    digigait_meta = digigait_reference.load_metadata(video_path)
    if digigait_meta is not None:
        print(f"[main] Found DigiGait output folder: {digigait_meta.images_dir.name} "
              f"-- using its exact calibration/belt-speed/orientation where available.")

    meta = read_video_metadata(video_path, interactive=interactive)
    if digigait_meta is not None and digigait_meta.belt_speed_cms is not None:
        meta.belt_speed_cms = digigait_meta.belt_speed_cms
        print(f"[main] Belt speed overridden from belt_speed.txt: {meta.belt_speed_cms} cm/s")

    first_frame = read_first_frame(video_path)

    if digigait_meta is not None and digigait_meta.calibration is not None:
        calibration = digigait_meta.calibration
        print(f"[main] Calibration from AOI_dimensions.txt: "
              f"{calibration.cm_per_pixel_x:.5f} cm/px (x), {calibration.cm_per_pixel_y:.5f} cm/px (y)")
    else:
        calibration = calibrate(first_frame, allow_interactive_fallback=interactive)

    if digigait_meta is not None and digigait_meta.orientation is not None:
        orientation = digigait_meta.orientation
        print(f"[main] Orientation from mouth-coordinates ground truth: "
              f"nose_direction='{orientation.nose_direction}'")
    else:
        orientation = resolve_orientation(first_frame if interactive else None)

    tracks = build_paw_tracks(video_path, meta, calibration, orientation)
    all_metrics = compute_all_metrics(tracks, meta, orientation)

    plot_paths = generate_all_plots(all_metrics, meta, config.PLOTS_DIR)
    print(f"[main] Plots saved: {[str(p) for p in plot_paths.values()]}")

    summary_df = build_summary_dataframe(all_metrics, meta)

    stem = video_path.stem
    summary_df.to_csv(config.CSV_DIR / f"{stem}_summary.csv", index=False)
    all_metrics["area_time_series"].to_csv(config.CSV_DIR / f"{stem}_area_time_series.csv", index=False)
    all_metrics["stance_width_df"].to_csv(config.CSV_DIR / f"{stem}_stance_width.csv", index=False)

    stride_rows = []
    for label, records in all_metrics["stride_records_by_paw"].items():
        for r in records:
            stride_rows.append({
                "paw": label,
                "onset_time_s": r.onset_time_s,
                "stride_time_s": r.stride_time_s,
                "stride_length_cm": r.stride_length_cm,
                "stance_duration_s": r.stance_duration_s,
                "swing_duration_s": r.swing_duration_s,
            })
    pd.DataFrame(stride_rows).to_csv(config.CSV_DIR / f"{stem}_strides.csv", index=False)

    print(f"[main] CSV exported to {config.CSV_DIR}")

    if digigait_meta is not None:
        indices_df = digigait_reference.load_reference_indices(digigait_meta.images_dir, stem)
        if indices_df is not None:
            comparison_df = digigait_reference.compare_summary_to_reference(summary_df, indices_df)
            if not comparison_df.empty:
                comparison_path = config.CSV_DIR / f"{stem}_vs_digigait_reference.csv"
                comparison_df.to_csv(comparison_path, index=False)
                print(f"[main] Reference comparison saved: {comparison_path}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Rat gait / momentum analysis pipeline")
    parser.add_argument("--video", type=str, default=None,
                         help="Path to a single video file. If omitted, processes every "
                              "video found in config.VIDEO_DIR.")
    parser.add_argument("--no-interactive", action="store_true",
                         help="Skip manual click/prompt steps (calibration/orientation/belt "
                              "speed must be resolvable from config/filename/auto-detection).")
    args = parser.parse_args()

    interactive = not args.no_interactive

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = discover_videos()
        if not videos:
            print(f"No videos found in {config.VIDEO_DIR}. "
                  f"Add .mp4/.avi/.mov files there, or pass --video.")
            return

    all_summaries = []
    for video_path in videos:
        try:
            all_summaries.append(process_video(video_path, interactive=interactive))
        except Exception as exc:
            print(f"[main] FAILED on {video_path.name}: {exc}")
            raise

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = config.CSV_DIR / "all_videos_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n[main] Combined summary for {len(all_summaries)} video(s): {combined_path}")


if __name__ == "__main__":
    main()
