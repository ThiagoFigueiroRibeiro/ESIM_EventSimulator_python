"""Convert a text event stream into a timestamped PNG frame sequence.

Example::

    python -m esim.event_frames demo_out/events.txt --output demo_out/event_frames
"""

import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from .viz import accumulate, to_rgb
from .writers import load_events_txt

NS_PER_MS = 1_000_000


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


def write_event_frames(
    events: np.ndarray,
    output: str,
    window_ns: int,
    shape: Tuple[int, int],
) -> int:
    """Group events into fixed time windows and write red/blue PNG frames.

    ON events are blue, OFF events are red, and pixels without events are white.
    An ``images.csv`` index is written alongside the numbered PNG files.
    """
    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    if not len(events):
        raise ValueError("no events to render")

    if len(events) > 1 and np.any(events["t"][1:] < events["t"][:-1]):
        events = np.sort(events, order="t", kind="stable")

    os.makedirs(output, exist_ok=True)
    first_window = (int(events["t"][0]) // window_ns) * window_ns
    frame_count = (int(events["t"][-1]) - first_window) // window_ns + 1
    entries = []

    for index in range(frame_count):
        start = first_window + index * window_ns
        end = start + window_ns
        left = int(np.searchsorted(events["t"], start, side="left"))
        right = int(np.searchsorted(events["t"], end, side="left"))
        accumulated = accumulate(events[left:right], shape=shape)
        rgb = np.rint(to_rgb(accumulated) * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        name = f"frame_{index:06d}.png"
        if not cv2.imwrite(os.path.join(output, name), bgr):
            raise IOError(f"could not write {name} to {output}")
        entries.append(f"{start},{name}")

    with open(os.path.join(output, "images.csv"), "w", encoding="utf-8") as handle:
        handle.write("# timestamp_ns, image\n")
        for entry in entries:
            handle.write(entry + "\n")
    return frame_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="esim.event_frames",
        description="Convert events.txt into red/blue event-frame PNGs.",
    )
    parser.add_argument("events", help="path to events.txt (or its containing folder)")
    parser.add_argument("-o", "--output", required=True, help="folder for the PNG sequence")
    parser.add_argument(
        "--window-ms",
        type=float,
        default=10.0,
        help="events accumulated per frame in milliseconds (default: 10)",
    )
    parser.add_argument("--width", type=int, default=None, help="sensor width (inferred by default)")
    parser.add_argument("--height", type=int, default=None, help="sensor height (inferred by default)")
    args = parser.parse_args(argv)

    if args.window_ms <= 0:
        parser.error("--window-ms must be positive")
    path = args.events
    if os.path.isdir(path):
        path = os.path.join(path, "events.txt")

    events = load_events_txt(path)
    shape = infer_shape(events, args.width, args.height)
    count = write_event_frames(
        events,
        args.output,
        int(round(args.window_ms * NS_PER_MS)),
        shape,
    )
    print(
        f"Wrote {count} event frames ({shape[1]}x{shape[0]}) "
        f"with {args.window_ms:g} ms windows to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
