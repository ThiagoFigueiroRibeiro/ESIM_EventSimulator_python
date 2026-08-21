"""Generate a synthetic image sequence for testing and demos.

Renders a translating sinusoidal pattern, which produces a predictable, dense event
stream: every vertical stripe edge fires ON events on one side and OFF on the other.

    python tools/make_test_sequence.py --output /tmp/seq --frames 200
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render(width: int, height: int, shift_px: float, period_px: float) -> np.ndarray:
    """One frame of a sinusoidal grating translated horizontally by ``shift_px``."""
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)

    phase = 2.0 * np.pi * (grid_x - shift_px) / period_px
    # A gentle vertical modulation keeps the pattern from being perfectly separable.
    vertical = 0.10 * np.sin(2.0 * np.pi * grid_y / (period_px * 3.0))
    # Amplitudes are chosen so intensity stays clear of 0: the log signal the event
    # model works on becomes extremely steep near black and would swamp the output.
    return np.clip(0.5 + 0.30 * np.sin(phase) + vertical, 0.05, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, help="folder to write the sequence into")
    parser.add_argument("--frames", type=int, default=200, help="number of frames")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--framerate", type=float, default=1000.0, help="frame rate in Hz")
    parser.add_argument("--speed", type=float, default=150.0, help="pattern speed in px/s")
    parser.add_argument("--period", type=float, default=30.0, help="grating period in px")
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    dt_ns = int(round(1e9 / args.framerate))

    entries = []
    for i in range(args.frames):
        t_s = i / args.framerate
        frame = render(args.width, args.height, args.speed * t_s, args.period)
        name = f"frame_{i:06d}.png"
        cv2.imwrite(os.path.join(args.output, name), (frame * 255.0).astype(np.uint8))
        entries.append(f"{i * dt_ns},{name}")

    with open(os.path.join(args.output, "images.csv"), "w", encoding="utf-8") as handle:
        handle.write("# timestamp_ns, image\n")
        for entry in entries:
            handle.write(entry + "\n")

    duration = args.frames / args.framerate
    print(
        f"Wrote {args.frames} frames ({args.width}x{args.height}) "
        f"spanning {duration:.3f} s to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
