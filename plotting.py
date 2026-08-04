"""
DigiGait-style plots, built from the dict returned by
metrics.compute_all_metrics().

Three plot types, matching the reference software:
  1. Dynamic Gait Signals -- 4-panel paw area vs time
  2. Ensemble Paws        -- normalized, averaged stride cycles per paw,
                             fore-pair and hind-pair, with N and dA/dT
  3. Posture Plot          -- 2D paw layout with stride/stance/angle
                             annotations

Styling (dark green background, bright per-paw line colors, legend N
values) is pulled from config so it's easy to retune without touching
plot logic.
"""

import matplotlib
# Non-interactive backend: these plots only ever get saved to disk, never
# shown in a window, and this file may run on a Flask worker thread
# (matplotlib's default macOS backend can only create figures on the
# main thread).
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

import config
from video_io import VideoMeta

PAW_ORDER = config.PAW_ORDER
COLORS = config.DIGIGAIT_LINE_COLORS
BG = config.DIGIGAIT_BG_COLOR


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, color="white", alpha=0.15, linewidth=0.5)


def plot_dynamic_gait_signals(all_metrics: dict, meta: VideoMeta, save_path):
    """4-panel plot: LF, RF, LH, RH paw contact area (cm^2) vs duration (s)."""
    df = all_metrics["area_time_series"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=BG)
    fig.suptitle(f"Dynamic Gait Signals -- {meta.path.name}", color="white", fontsize=13)

    for ax, label in zip(axes.flat, PAW_ORDER):
        _style_axes(ax)
        ax.plot(df["time_s"], df[label], color=COLORS[label], linewidth=1.2, label=label)
        ax.set_xlabel("Duration (sec)")
        ax.set_ylabel("Paw Area (cm$^2$)")
        ax.set_title(label, color=COLORS[label], fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", facecolor=BG, labelcolor="white", framealpha=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, facecolor=BG, dpi=150)
    plt.close(fig)


def plot_ensemble_paws(all_metrics: dict, meta: VideoMeta, save_path):
    """
    Averaged, normalized stride-cycle curves, one subplot for the fore
    pair (LF+RF) and one for the hind pair (LH+RH), each paw overlaid.
    Annotates N (stride count) and MIN/MAX dA/dT per paw.
    """
    ensembles = all_metrics["ensemble_by_paw"]
    dadt_summary = all_metrics["dadt_summary"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=BG)
    fig.suptitle(f"Ensemble Paws -- {meta.path.name}", color="white", fontsize=13)

    pairs = [("Fore", ["LF", "RF"]), ("Hind", ["LH", "RH"])]
    for ax, (pair_name, labels) in zip(axes, pairs):
        _style_axes(ax)
        legend_labels = []
        for label in labels:
            ens = ensembles[label]
            pct = ens["pct_axis"]
            mean_curve = ens["mean_curve"]
            std_curve = ens["std_curve"]
            color = COLORS[label]

            ax.plot(pct, mean_curve, color=color, linewidth=1.8)
            ax.fill_between(pct, mean_curve - std_curve, mean_curve + std_curve,
                             color=color, alpha=0.2)

            dadt_min, dadt_max = dadt_summary[label]
            legend_labels.append(
                f"{label}  N={ens['n_strides']}  "
                f"dA/dT min={dadt_min:.2f} max={dadt_max:.2f} cm$^2$/s"
            )

        ax.set_xlabel("Stride Cycle (%)")
        ax.set_ylabel("Paw Area (cm$^2$)")
        ax.set_title(f"{pair_name} Pair", color="white", fontsize=11)
        ax.legend(legend_labels, loc="upper right", facecolor=BG, labelcolor="white",
                   framealpha=0.5, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, facecolor=BG, dpi=150)
    plt.close(fig)


def plot_posture(all_metrics: dict, meta: VideoMeta, save_path):
    """
    2D layout of average paw placement positions (LF/RF/LH/RH), with
    stride length, paw angle, stance width, and step angle listed
    alongside -- mirrors DigiGait's Posture Plot.
    """
    stance_runs_by_paw = all_metrics["stance_runs_by_paw"]
    stride_records_by_paw = all_metrics["stride_records_by_paw"]
    step_angles_by_paw = all_metrics["step_angles_by_paw"]
    stance_width_df = all_metrics["stance_width_df"]

    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, figsize=(12, 6), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.15]}
    )
    fig.suptitle(f"Posture Plot -- {meta.path.name}", color="white", fontsize=13)
    _style_axes(ax_plot)
    ax_plot.set_xlabel("X (cm)")
    ax_plot.set_ylabel("Y (cm)")
    ax_plot.set_aspect("equal")
    # Pixel-space y grows downward; invert so "up" in the plot matches
    # "up" (toward the nose) in the source video frame.
    ax_plot.invert_yaxis()

    lines_out = []
    for label in PAW_ORDER:
        runs = stance_runs_by_paw[label]
        if not runs:
            continue
        xs = [r.centroid_cm[0] for r in runs]
        ys = [r.centroid_cm[1] for r in runs]
        ax_plot.scatter(xs, ys, color=COLORS[label], s=40, label=label, zorder=3)
        ax_plot.plot(xs, ys, color=COLORS[label], alpha=0.4, linewidth=1, zorder=2)

        strides = stride_records_by_paw[label]
        mean_stride_len = np.mean([s.stride_length_cm for s in strides]) if strides else float("nan")
        # Paw placement angle: ellipse major-axis angle at each stance
        # phase's peak-loading (max area) frame only, averaged across phases.
        mean_placement_angle = np.mean([r.paw_placement_angle_deg for r in runs])
        step_angles = step_angles_by_paw[label]
        mean_step_angle = np.mean(step_angles) if step_angles else float("nan")

        lines_out.append(
            f"{label}:  stride length = {mean_stride_len:.2f} cm\n"
            f"      placement angle = {mean_placement_angle:.1f} deg   "
            f"step angle = {mean_step_angle:.1f} deg"
        )

    # Legend placed above the axes (not inside the data area) so it never
    # sits on top of a paw's actual plotted position, and below the
    # figure-level title.
    ax_plot.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=4,
                    facecolor=BG, labelcolor="white", framealpha=0.5)

    fore_width = stance_width_df["fore_width_cm"].mean()
    hind_width = stance_width_df["hind_width_cm"].mean()
    lines_out.append("")
    lines_out.append(f"Fore stance width (LF-RF): {fore_width:.2f} cm")
    lines_out.append(f"Hind stance width (LH-RH): {hind_width:.2f} cm")
    lines_out.append(f"Step sequence regularity: {all_metrics['regularity_pct']:.1f}%")
    lines_out.append(f"Animal length (proxy): {all_metrics['animal_dims']['animal_length_cm']:.2f} cm")
    lines_out.append(f"Animal width (proxy): {all_metrics['animal_dims']['animal_width_cm']:.2f} cm")

    ax_text.set_facecolor(BG)
    ax_text.axis("off")
    ax_text.text(
        0.02, 0.98, "\n".join(lines_out),
        transform=ax_text.transAxes, va="top", ha="left",
        color="white", fontsize=9.5, family="monospace", wrap=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, facecolor=BG, dpi=150)
    plt.close(fig)


def generate_all_plots(all_metrics: dict, meta: VideoMeta, plots_dir):
    """Generate all three DigiGait-style plots for one video and save to disk."""
    stem = meta.path.stem
    paths = {
        "gait_signals": plots_dir / f"{stem}_dynamic_gait_signals.png",
        "ensemble_paws": plots_dir / f"{stem}_ensemble_paws.png",
        "posture": plots_dir / f"{stem}_posture_plot.png",
    }
    plot_dynamic_gait_signals(all_metrics, meta, paths["gait_signals"])
    plot_ensemble_paws(all_metrics, meta, paths["ensemble_paws"])
    plot_posture(all_metrics, meta, paths["posture"])
    return paths
