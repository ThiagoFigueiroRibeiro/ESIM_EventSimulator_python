"""Convert an event stream into a timestamped PNG frame sequence.

Example::

    python -m esim.event_frames demo_out/events.npz --output demo_out/event_frames
"""

import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np

NS_PER_MS = 1_000_000


def load_events_npz(path: str) -> np.ndarray:
    """Load events from an .npz file as a structured NumPy array.

    Supports either:
      - a structured array stored under the key ``events``
      - separate arrays ``x``, ``y``, ``t``, and ``pol`` (preferred)
      - separate arrays ``x``, ``y``, ``t``, and ``p`` (legacy support)

    The returned array always uses ``pol`` as the polarity field name.
    """
    with np.load(path, allow_pickle=False) as data:
        # Case 1: structured array under "events"
        if "events" in data.files:
            events = data["events"]
            if events.dtype.names is None:
                raise ValueError(f"{path} contains 'events' but it is not a structured array")

            if "pol" in events.dtype.names:
                return events

            if "p" in events.dtype.names:
                out = np.empty(
                    len(events),
                    dtype=[
                        ("x", events["x"].dtype),
                        ("y", events["y"].dtype),
                        ("t", events["t"].dtype),
                        ("pol", events["p"].dtype),
                    ],
                )
                out["x"] = events["x"]
                out["y"] = events["y"]
                out["t"] = events["t"]
                out["pol"] = events["p"]
                return out

            raise ValueError(
                f"{path} contains structured 'events' but no 'pol' or 'p' field"
            )

        # Case 2: separate arrays
        required = ("x", "y", "t")
        if all(name in data.files for name in required) and ("pol" in data.files or "p" in data.files):
            x = data["x"]
            y = data["y"]
            t = data["t"]
            pol = data["pol"] if "pol" in data.files else data["p"]

            if not (len(x) == len(y) == len(t) == len(pol)):
                raise ValueError(f"{path} contains x/y/t/pol arrays of mismatched lengths")

            events = np.empty(
                len(x),
                dtype=[
                    ("x", x.dtype),
                    ("y", y.dtype),
                    ("t", t.dtype),
                    ("pol", pol.dtype),
                ],
            )
            events["x"] = x
            events["y"] = y
            events["t"] = t
            events["pol"] = pol
            return events

        raise ValueError(
            f"{path} must contain either a structured 'events' array or arrays x, y, t, pol; "
            f"found {data.files}"
        )


def infer_shape(events: np.ndarray, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    """Return ``(height, width)``, inferring unspecified dimensions from events."""
    if not len(events):
        raise ValueError("cannot infer sensor size from an empty event stream")

    minimum_width = int(events["x"].max()) + 1
    minimum_height = int(events["y"].max()) + 1

    width = minimum_width if width is None else width
    height = minimum_height if height is None else height

    if width < minimum_width or height < minimum_height:
        raise ValueError(
            f"sensor size {width}x{height} is smaller than event coordinates "
            f"({minimum_width}x{minimum_height} required)"
        )

    return height, width


def accumulate_window(
    flat_indices: np.ndarray,
    weights: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Accumulate events into a 2D image using np.bincount.

    Uses +1 for positive polarity and -1 for negative polarity.
    """
    height, width = shape
    if flat_indices.size == 0:
        return np.zeros(shape, dtype=np.float32)

    acc = np.bincount(flat_indices, weights=weights, minlength=height * width)
    return acc.reshape(shape)


def write_event_frames(
    events: np.ndarray,
    output: str,
    window_ns: int,
    shape: Tuple[int, int],
    assume_sorted: bool = False,
    png_compression: int = 0,
) -> int:
    """Group events into fixed time windows and write PNG frames."""
    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    if not len(events):
        raise ValueError("no events to render")

    def hex_rgb(value: str) -> np.ndarray:
        value = value.lstrip("#")
        return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)

    # Palette defined in RGB, then converted to BGR for OpenCV.
    background_bgr = hex_rgb("#1E2636")[::-1].copy()
    positive_bgr = hex_rgb("#FAFFFF")[::-1].copy()
    negative_bgr = hex_rgb("#4F7BB6")[::-1].copy()

    background_u8 = np.rint(background_bgr).astype(np.uint8)
    pos_delta = positive_bgr - background_bgr
    neg_delta = negative_bgr - background_bgr

    # Pull fields into separate arrays for better locality and faster processing.
    x = np.asarray(events["x"])
    y = np.asarray(events["y"])
    t = np.asarray(events["t"])
    pol = np.asarray(events["pol"])

    # Sort once if needed. This is much cheaper than per-frame binary search.
    if not assume_sorted and len(t) > 1 and np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        x = x[order]
        y = y[order]
        t = t[order]
        pol = pol[order]

    # Precompute flat pixel indices and polarity weights once.
    height, width = shape
    flat_indices = y.astype(np.int64, copy=False) * width + x.astype(np.int64, copy=False)
    weights = np.where(pol > 0, 1, -1).astype(np.int8, copy=False)

    os.makedirs(output, exist_ok=True)

    first_window = (int(t[0]) // window_ns) * window_ns
    frame_count = (int(t[-1]) - first_window) // window_ns + 1

    # Reuse the frame buffer for each output image.
    frame = np.empty((height, width, 3), dtype=np.uint8)
    flat_frame = frame.reshape(-1, 3)

    csv_path = os.path.join(output, "images.csv")
    with open(csv_path, "w", encoding="utf-8") as csv_handle:
        csv_handle.write("# timestamp_ns, image\n")

        left = 0
        right = 0
        n = len(t)

        for index in range(frame_count):
            start = first_window + index * window_ns
            end = start + window_ns

            # One-pass monotonic boundary advancement.
            while left < n and t[left] < start:
                left += 1
            while right < n and t[right] < end:
                right += 1

            accumulated = accumulate_window(
                flat_indices[left:right],
                weights[left:right],
                shape,
            )

            # Start with the background.
            frame[:] = background_u8

            values = np.asarray(accumulated)
            flat_values = values.reshape(-1)

            active_idx = np.flatnonzero(flat_values)
            if active_idx.size:
                active_vals = flat_values[active_idx].astype(np.float32, copy=False)
                max_abs = float(np.max(np.abs(active_vals)))

                if max_abs > 0.0:
                    strengths = np.abs(active_vals) / max_abs

                    # Positive pixels use the positive delta; negative pixels use the negative delta.
                    deltas = np.where(active_vals[:, None] > 0, pos_delta, neg_delta)
                    colors = background_bgr + strengths[:, None] * deltas
                    flat_frame[active_idx] = np.rint(colors).astype(np.uint8)

            name = f"frame_{index:06d}.png"
            path = os.path.join(output, name)
            if not cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]):
                raise IOError(f"could not write {name} to {output}")

            csv_handle.write(f"{start},{name}\n")

    return frame_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="esim.event_frames",
        description="Convert events.npz into red/blue event-frame PNGs.",
    )
    parser.add_argument("events", help="path to events.npz (or its containing folder)")
    parser.add_argument("-o", "--output", required=True, help="folder for the PNG sequence")

    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--window-ms",
        type=float,
        default=10.0,
        help="events accumulated per frame in milliseconds (default: 10)",
    )
    time_group.add_argument(
        "--fps",
        type=float,
        default=None,
        help="video FPS; window size becomes half a frame period: 1000 / (2 * fps) ms",
    )

    parser.add_argument("--width", type=int, default=None, help="sensor width (inferred by default)")
    parser.add_argument("--height", type=int, default=None, help="sensor height (inferred by default)")
    parser.add_argument(
        "--assume-sorted",
        action="store_true",
        help="skip timestamp sorting/checking; use only if events are already sorted by t",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=0,
        help="PNG compression level (0 fastest, 9 smallest; default: 0)",
    )
    args = parser.parse_args(argv)

    if args.fps is not None:
        if args.fps <= 0:
            parser.error("--fps must be positive")
        window_ms = 1000.0 / (2.0 * args.fps)
    else:
        if args.window_ms <= 0:
            parser.error("--window-ms must be positive")
        window_ms = args.window_ms

    if not (0 <= args.png_compression <= 9):
        parser.error("--png-compression must be in range 0..9")

    path = args.events
    if os.path.isdir(path):
        path = os.path.join(path, "events.npz")

    events = load_events_npz(path)
    shape = infer_shape(events, args.width, args.height)

    count = write_event_frames(
        events,
        args.output,
        int(round(window_ms * NS_PER_MS)),
        shape,
        assume_sorted=args.assume_sorted,
        png_compression=args.png_compression,
    )

    if args.fps is not None:
        print(
            f"Wrote {count} event frames ({shape[1]}x{shape[0]}) "
            f"with fps={args.fps:g} -> window={window_ms:g} ms to {args.output}"
        )
    else:
        print(
            f"Wrote {count} event frames ({shape[1]}x{shape[0]}) "
            f"with {window_ms:g} ms windows to {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())