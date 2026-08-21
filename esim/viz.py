"""Visualisation of a generated event stream.

Run as a script to inspect a result folder::

    python -m esim.viz output/events.npz
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np

from .writers import load_events_npz

NS_PER_S = 1_000_000_000


def accumulate(
    events: np.ndarray,
    shape: Optional[Tuple[int, int]] = None,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> np.ndarray:
    """Sum event polarities per pixel over a time window.

    Returns:
        A 2D array where positive values mean more ON than OFF events.
    """
    if t_start is not None:
        events = events[events["t"] >= t_start]
    if t_end is not None:
        events = events[events["t"] < t_end]

    if shape is None:
        if not len(events):
            raise ValueError("cannot infer sensor size from an empty event array")
        shape = (int(events["y"].max()) + 1, int(events["x"].max()) + 1)

    frame = np.zeros(shape, dtype=np.int32)
    np.add.at(
        frame,
        (events["y"].astype(np.intp), events["x"].astype(np.intp)),
        np.where(events["pol"], 1, -1),
    )
    return frame


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """Render an accumulated frame as red (OFF) / blue (ON) on white."""
    peak = max(1, int(np.abs(frame).max()))
    normalized = np.clip(frame / peak, -1.0, 1.0)

    rgb = np.ones(frame.shape + (3,), dtype=np.float64)
    on = normalized > 0
    off = normalized < 0
    # Fade the other two channels towards the event colour.
    rgb[..., 0][on] = 1.0 - normalized[on]
    rgb[..., 1][on] = 1.0 - normalized[on]
    rgb[..., 1][off] = 1.0 + normalized[off]
    rgb[..., 2][off] = 1.0 + normalized[off]
    return rgb


def plot(events: np.ndarray, shape=None, bins: int = 100, save_to: Optional[str] = None) -> None:
    """Show the accumulated event image alongside the event rate over time."""
    import matplotlib

    if save_to:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not len(events):
        raise ValueError("no events to plot")

    fig, (ax_img, ax_rate) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_img.imshow(to_rgb(accumulate(events, shape)), interpolation="nearest")
    ax_img.set_title(f"Accumulated events (n = {len(events):,})")
    ax_img.set_xlabel("x [px]")
    ax_img.set_ylabel("y [px]")

    times_s = events["t"] / NS_PER_S
    counts, edges = np.histogram(times_s, bins=bins)
    width = edges[1] - edges[0]
    ax_rate.plot(edges[:-1], counts / width if width > 0 else counts)
    ax_rate.set_title("Event rate")
    ax_rate.set_xlabel("time [s]")
    ax_rate.set_ylabel("events / s")
    ax_rate.grid(alpha=0.3)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="esim.viz", description="Visualise an event stream produced by esim."
    )
    parser.add_argument("events", help="path to events.npz (or the folder containing it)")
    parser.add_argument("--bins", type=int, default=100, help="histogram bins for the rate plot")
    parser.add_argument("--save-to", default=None, help="write a PNG instead of opening a window")
    args = parser.parse_args(argv)

    path = args.events
    if os.path.isdir(path):
        path = os.path.join(path, "events.npz")

    plot(load_events_npz(path), bins=args.bins, save_to=args.save_to)
    if args.save_to:
        print(f"Wrote {args.save_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
