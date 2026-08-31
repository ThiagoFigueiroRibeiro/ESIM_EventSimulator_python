#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="events.npz", help="Path to events.npz")

    parser.add_argument("--fps", type=int, required=True,
                        help="Number of equally spaced time bins/frames (exactly this many frames).")
    parser.add_argument("--first_events", type=int, default=50,
                        help="How many first events to plot per time bin.")
    parser.add_argument("--baricenter_frames", type=str, default="./baricenter_frames",
                        help="Output directory for frame images.")

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--image_ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--frame_prefix", type=str, default="frame")

    parser.add_argument("--plot_events", action="store_true",
                        help="If set, uses event scatter. Otherwise, only barycenter (not recommended).")
    parser.add_argument("--events_marker_size", type=float, default=6.0)
    parser.add_argument("--bary_marker_size", type=float, default=80.0)

    parser.add_argument("--show", action="store_true", help="Show figures while generating (slow).")
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.first_events <= 0:
        raise ValueError("--first_events must be > 0")

    time, x, y, keys = load_events(args.npz)

    os.makedirs(args.baricenter_frames, exist_ok=True)

    first_ts = float(time[0])
    last_ts = float(time[-1])
    edges = np.linspace(first_ts, last_ts, args.fps + 1)

    # Fixed axis limits for consistent visuals across frames
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_pad = 0.02 * (x_max - x_min + 1e-12)
    y_pad = 0.02 * (y_max - y_min + 1e-12)
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)

    print(f"Loaded keys: time={keys[0]}, x={keys[1]}, y={keys[2]}")
    print(f"Saving {args.fps} frames to: {args.baricenter_frames}")

    for i in range(args.fps):
        left = edges[i]
        right = edges[i + 1]

        idx_start = np.searchsorted(time, left, side="left")
        if i < args.fps - 1:
            idx_end = np.searchsorted(time, right, side="left")
        else:
            idx_end = np.searchsorted(time, right, side="right")  # include last_ts

        # Take first N events within this bin
        take_end = min(idx_start + args.first_events, idx_end)

        fig, ax = plt.subplots(figsize=(6, 6))

        if take_end <= idx_start:
            # No events in this bin
            ax.set_title(f"bin {i:04d}\nt=[{left:.6g}, {right:.6g}] (no events)")
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_ylim(y_lim[1], y_lim[0])
            ax.set_aspect("equal", adjustable="box")
        else:
            xs = x[idx_start:take_end]
            ys = y[idx_start:take_end]

            # Barycenter of plotted events
            xb = float(np.mean(xs))
            yb = float(np.mean(ys))

            # Plot events
            if args.plot_events:
                ax.scatter(xs, ys, s=args.events_marker_size, c="blue", alpha=0.7, linewidths=0)

            # Plot barycenter
            ax.scatter([xb], [yb], s=args.bary_marker_size, c="red", alpha=1.0, linewidths=0)

            t_first_event = float(time[idx_start])
            ax.set_title(f"bin {i:04d}\nt_first={t_first_event:.6g}\nN={take_end - idx_start}")

            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_ylim(y_lim[1], y_lim[0])
            ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        out_path = os.path.join(args.baricenter_frames, f"{args.frame_prefix}_{i:04d}.{args.image_ext}")
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