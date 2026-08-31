#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def find_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None


def load_events(npz_path):
    npz = np.load(npz_path)

    time_key = find_key(npz, ["time", "t", "timestamp", "timestamps"])
    x_key = find_key(npz, ["x"])
    y_key = find_key(npz, ["y"])

    if time_key is None or x_key is None or y_key is None:
        keys = list(npz.keys())
        raise KeyError(
            f"Could not find required keys in {npz_path}.\n"
            f"Found keys: {keys}\n"
            f"Expected something like time/t/timestamp and x,y."
        )

    time = np.asarray(npz[time_key])
    x = np.asarray(npz[x_key])
    y = np.asarray(npz[y_key])

    if time.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected time, x, y to be 1D arrays.")
    if not (len(time) == len(x) == len(y)):
        raise ValueError("Expected time, x, y arrays to have the same length.")

    if len(time) >= 2 and np.any(time[1:] < time[:-1]):
        raise ValueError("Timestamps appear not to be sorted. This script assumes sorted by time.")

    return time, x, y, (time_key, x_key, y_key)


def pca_2d(x, y):
    """
    Returns:
      mu (2,), vals (2,), vecs (2,2)

    vecs[:, 0] = PC1 direction
    vecs[:, 1] = PC2 direction
    vals sorted descending
    """
    pts = np.column_stack([x, y])
    mu = pts.mean(axis=0)
    xc = pts - mu

    C = (xc.T @ xc) / max(len(pts), 1)
    vals, vecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(vals)[::-1]   # descending
    vals = vals[order]
    vecs = vecs[:, order]
    return mu, vals, vecs


def pca_center_2d(x, y):
    """
    Compute a PCA-based center:
      1) project points into PCA coordinates
      2) take midpoint of the min/max extent along PC1 and PC2
      3) map back to original coordinates

    This gives a center aligned with the PCA axes, instead of the barycenter.
    """
    pts = np.column_stack([x, y])
    mu, vals, vecs = pca_2d(x, y)

    # Coordinates in PCA basis
    coords = (pts - mu) @ vecs  # shape: (N, 2)

    # Midpoint of the extent in PCA coordinates
    center_pc = np.array([
        0.5 * (coords[:, 0].min() + coords[:, 0].max()),  # along PC1
        0.5 * (coords[:, 1].min() + coords[:, 1].max()),  # along PC2
    ])

    # Back to original space
    center = mu + center_pc @ vecs.T
    return center, mu, vals, vecs


def get_bin_slice(time, edges, bin_idx, fps):
    left = edges[bin_idx]
    right = edges[bin_idx + 1]

    idx_start = np.searchsorted(time, left, side="left")
    if bin_idx < fps - 1:
        idx_end = np.searchsorted(time, right, side="left")
    else:
        idx_end = np.searchsorted(time, right, side="right")  # include last_ts
    return idx_start, idx_end


def collect_two_bin_events(time, x, y, edges, fps, i, first_events):
    """
    Collect first first_events events from two consecutive bins:
      frame i uses bins i and i+1
    For the last frame, reuse the last two bins to preserve frame count.
    """
    if fps < 2:
        raise ValueError("Need at least 2 bins to use two consecutive timeframes.")

    if i < fps - 1:
        b0, b1 = i, i + 1
    else:
        b0, b1 = fps - 2, fps - 1

    xs_all = []
    ys_all = []
    ts_all = []

    for b in [b0, b1]:
        idx_start, idx_end = get_bin_slice(time, edges, b, fps)
        take_end = min(idx_start + first_events, idx_end)
        if take_end > idx_start:
            xs_all.append(x[idx_start:take_end])
            ys_all.append(y[idx_start:take_end])
            ts_all.append(time[idx_start:take_end])

    if len(xs_all) == 0:
        return b0, b1, np.array([]), np.array([]), np.array([])

    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    ts = np.concatenate(ts_all)
    return b0, b1, xs, ys, ts


def draw_single_arrow(ax, p0, p1, color, lw=3, mutation_scale=14):
    """
    Draw a single-headed arrow from p0 to p1.
    """
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="->",
        color=color,
        linewidth=lw,
        mutation_scale=mutation_scale,
        shrinkA=0,
        shrinkB=0,
        clip_on=False
    )
    ax.add_patch(arrow)
    return arrow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="events.npz", help="Path to events.npz")
    parser.add_argument("--fps", type=int, required=True,
                        help="Number of equally spaced time bins/frames.")
    parser.add_argument("--first_events", type=int, default=50,
                        help="How many first events to take from each of the two bins.")
    parser.add_argument("--PCA_frames", type=str, default="./PCA_frames",
                        help="Output directory for frame images.")

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--image_ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--frame_prefix", type=str, default="frame")

    parser.add_argument("--plot_events", action="store_true", default=True,
                        help="Plot events in the frame.")
    parser.add_argument("--no_plot_events", dest="plot_events", action="store_false",
                        help="Disable event plotting.")
    parser.add_argument("--events_marker_size", type=float, default=8.0)

    parser.add_argument("--min_points_for_pca", type=int, default=3,
                        help="If fewer than this number of points exist, PCA is skipped.")

    parser.add_argument("--pca_len_mode", type=str, default="sqrt_eig",
                        choices=["range", "sqrt_eig"],
                        help="Scaling mode for PCA arrows.")
    parser.add_argument("--pca_len_scale", type=float, default=0.8,
                        help="Global multiplicative factor for PCA arrow lengths.")

    parser.add_argument("--pc1_boost", type=float, default=2.5,
                        help="Extra multiplier for PC1 (red) arrow length.")
    parser.add_argument("--pc2_boost", type=float, default=1.0,
                        help="Extra multiplier for PC2 (green) arrow length.")

    parser.add_argument("--arrow_mutation_scale", type=float, default=18.0,
                        help="Arrowhead size.")
    parser.add_argument("--arrow_lw", type=float, default=3.0,
                        help="Arrow line width.")

    parser.add_argument("--show", action="store_true", help="Show figures while generating (slow).")
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.first_events <= 0:
        raise ValueError("--first_events must be > 0")

    time, x, y, keys = load_events(args.npz)

    os.makedirs(args.PCA_frames, exist_ok=True)

    first_ts = float(time[0])
    last_ts = float(time[-1])
    edges = np.linspace(first_ts, last_ts, args.fps + 1)

    # Fixed axis limits for consistent visuals
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_pad = 0.02 * (x_max - x_min + 1e-12)
    y_pad = 0.02 * (y_max - y_min + 1e-12)
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)

    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]

    # To keep arrow direction consistent across frames
    prev_v1 = None
    prev_v2 = None

    print(f"Loaded keys: time={keys[0]}, x={keys[1]}, y={keys[2]}")
    print(f"Saving {args.fps} frames to: {args.PCA_frames}")

    for i in range(args.fps):
        b0, b1, xs, ys, ts = collect_two_bin_events(
            time, x, y, edges, args.fps, i, args.first_events
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        if len(xs) < args.min_points_for_pca:
            ax.set_title(f"frame {i:04d}\nbins {b0} & {b1}\n(not enough points)")
            if args.plot_events and len(xs) > 0:
                ax.scatter(xs, ys, s=args.events_marker_size, c="blue", alpha=0.7, linewidths=0)

            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            # PCA center and axes
            center, mu, vals, vecs = pca_center_2d(xs, ys)
            cx, cy = float(center[0]), float(center[1])

            v1 = vecs[:, 0].copy()  # PC1
            v2 = vecs[:, 1].copy()  # PC2

            # Stabilize sign across frames so arrows don't randomly flip direction
            if prev_v1 is not None and np.dot(v1, prev_v1) < 0:
                v1 = -v1
            if prev_v2 is not None and np.dot(v2, prev_v2) < 0:
                v2 = -v2

            prev_v1 = v1.copy()
            prev_v2 = v2.copy()

            if args.pca_len_mode == "range":
                base = max(x_range, y_range)
                half_len_pc1 = args.pca_len_scale * base * 0.5 * args.pc1_boost
                half_len_pc2 = args.pca_len_scale * base * 0.5 * args.pc2_boost
            else:
                # Dynamic length per frame based on the local spread
                half_len_pc1 = args.pca_len_scale * float(np.sqrt(vals[0] + 1e-12)) * args.pc1_boost
                half_len_pc2 = args.pca_len_scale * float(np.sqrt(vals[1] + 1e-12)) * args.pc2_boost

            if args.plot_events:
                ax.scatter(xs, ys, s=args.events_marker_size, c="blue", alpha=0.65, linewidths=0)

            # PCA-based center point
            ax.scatter([cx], [cy], c="black", s=30, alpha=0.9, linewidths=0, zorder=5)

            # PC1: one arrow from center
            p1_start = (cx, cy)
            p1_end = (cx + half_len_pc1 * v1[0], cy + half_len_pc1 * v1[1])
            draw_single_arrow(
                ax, p1_start, p1_end,
                color="red",
                lw=args.arrow_lw,
                mutation_scale=args.arrow_mutation_scale
            )

            # PC2: one arrow from center
            p2_start = (cx, cy)
            p2_end = (cx + half_len_pc2 * v2[0], cy + half_len_pc2 * v2[1])
            draw_single_arrow(
                ax, p2_start, p2_end,
                color="green",
                lw=args.arrow_lw,
                mutation_scale=args.arrow_mutation_scale
            )

            t_first = float(ts[0]) if len(ts) > 0 else np.nan
            ax.set_title(
                f"bins {b0} & {b1}\n"
                f"t_first={t_first:.6g}\n"
                f"N={len(xs)}, PCA center, PC1(red), PC2(green)"
            )

            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        out_path = os.path.join(
            args.PCA_frames,
            f"{args.frame_prefix}_{i:04d}.{args.image_ext}"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.dpi)

        if args.show:
            plt.show()

        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)