"""
Command line entry point: turn an image sequence into an event stream.

Replaces the gflags-driven ``esim_node`` of the C++ version. Arguments can be kept
in a file and passed with ``@``, one flag per line, which mirrors how the original
flagfiles worked::

    python -m esim.cli @cfg/example.conf
"""

import argparse
import os
import sys
import tempfile
import time as wallclock
from typing import List, Optional

import numpy as np

from .camera_simulator import CameraSimulator
from .data_provider import FolderImageSource
from .event_simulator import EventSimulator
from .types import EventSimConfig, empty_events
from .writers import ImageSequenceWriter, concatenate_events, save_events_npz, save_events_txt

NS_PER_S = 1_000_000_000
MAX_EVENTS_PER_SHARD = 50_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esim",
        description="Simulate an event camera from a folder of stamped images.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="folder containing images.csv and the images"
    )
    parser.add_argument("-o", "--output", required=True, help="folder to write results into")

    events = parser.add_argument_group("event generation")
    events.add_argument(
        "--contrast-threshold",
        type=float,
        default=None,
        help="set both C+ and C- at once (overrides --contrast-threshold-pos/neg)",
    )
    events.add_argument("--contrast-threshold-pos", type=float, default=1.0, help="C+")
    events.add_argument("--contrast-threshold-neg", type=float, default=1.0, help="C-")
    # The C++ gflags default is 0.021, but every configuration ESIM ships sets these
    # to 0. Threshold noise starves the event stream when the per-frame intensity step
    # is much smaller than the noise itself, so 0 is the more useful default here.
    events.add_argument(
        "--contrast-threshold-sigma-pos", type=float, default=0.0, help="noise sigma on C+"
    )
    events.add_argument(
        "--contrast-threshold-sigma-neg", type=float, default=0.0, help="noise sigma on C-"
    )
    events.add_argument(
        "--refractory-period-ns",
        type=int,
        default=0,
        help="minimum time between two events at the same pixel",
    )
    events.add_argument(
        "--no-log-image",
        action="store_true",
        help="apply the thresholds to raw intensity instead of log intensity",
    )
    events.add_argument("--log-eps", type=float, default=0.001, help="epsilon added before log")
    events.add_argument("--random-seed", type=int, default=None, help="seed for threshold noise")

    camera = parser.add_argument_group("frame camera")
    camera.add_argument(
        "--exposure-time-ms",
        type=float,
        default=10.0,
        help="exposure time used to synthesise motion blur",
    )
    camera.add_argument(
        "--no-blurred-frames", action="store_true", help="skip motion-blurred frame output"
    )

    out = parser.add_argument_group("output")
    out.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def config_from_args(args: argparse.Namespace) -> EventSimConfig:
    cp = args.contrast_threshold if args.contrast_threshold is not None else args.contrast_threshold_pos
    cm = args.contrast_threshold if args.contrast_threshold is not None else args.contrast_threshold_neg
    return EventSimConfig(
        Cp=cp,
        Cm=cm,
        sigma_Cp=args.contrast_threshold_sigma_pos,
        sigma_Cm=args.contrast_threshold_sigma_neg,
        refractory_period_ns=args.refractory_period_ns,
        use_log_image=not args.no_log_image,
        log_eps=args.log_eps,
        random_seed=args.random_seed,
    )


def run(args: argparse.Namespace) -> int:
    source = FolderImageSource(args.input)
    config = config_from_args(args)
    simulator = EventSimulator(config)
    camera = None if args.no_blurred_frames else CameraSimulator(args.exposure_time_ms)

    os.makedirs(args.output, exist_ok=True)
    frame_writer = (
        None if camera is None else ImageSequenceWriter(os.path.join(args.output, "frames"))
    )

    if not args.quiet:
        print(f"Reading {len(source)} images from {args.input}")
        print(f"C+ = {config.Cp}, C- = {config.Cm}, log image = {config.use_log_image}")

    started = wallclock.perf_counter()
    n_frames = 0
    last_stamp = 0
    total_events = 0
    processed_images = 0

    # Temporary shard storage under the output folder
    with tempfile.TemporaryDirectory(prefix="events_shards_", dir=args.output) as tmp_dir:
        shard_paths: List[str] = []

        # Events buffered for the current shard
        pending: List[np.ndarray] = []
        pending_count = 0

        def flush_pending() -> None:
            """Write the current buffered events to a temporary shard file."""
            nonlocal pending, pending_count
            if not pending:
                return

            shard_events = concatenate_events(pending)
            shard_path = os.path.join(tmp_dir, f"events_{len(shard_paths):06d}.npz")
            save_events_npz(shard_path, shard_events)

            shard_paths.append(shard_path)
            pending = []
            pending_count = 0

        def add_events(chunk: np.ndarray) -> None:
            """
            Add an event chunk to the current shard buffer.

            If a single chunk would exceed MAX_EVENTS_PER_SHARD, split it across
            multiple shards.
            """
            nonlocal pending_count, total_events

            if len(chunk) == 0:
                return

            start = 0
            while start < len(chunk):
                room = MAX_EVENTS_PER_SHARD - pending_count
                take = min(room, len(chunk) - start)

                pending.append(chunk[start : start + take])
                pending_count += take
                total_events += take
                start += take

                if pending_count >= MAX_EVENTS_PER_SHARD:
                    flush_pending()

        try:
            for stamp, image in source:
                ev = simulator.image_callback(image, stamp)
                add_events(ev)
                last_stamp = stamp
                processed_images += 1

                if camera is not None:
                    blurred = camera.image_callback(image, stamp)
                    if blurred is not None:
                        frame_writer.add(blurred, camera.mid_exposure_time(stamp))
                        n_frames += 1

                if not args.quiet and processed_images % 100 == 0:
                    print(f"  {processed_images}/{len(source)} images", end="\r", flush=True)

            # Flush the last partial shard
            flush_pending()

            # Concatenate all shard files into the final output
            npz_path = os.path.join(args.output, "events.npz")
            if len(shard_paths) == 0:
                save_events_npz(npz_path, empty_events())
            elif len(shard_paths) == 1:
                # Avoid an unnecessary copy if there is only one shard
                os.replace(shard_paths[0], npz_path)
            else:
                parts = []
                for p in shard_paths:
                    with np.load(p) as data:
                        # Use the first array stored in the npz, regardless of key name
                        parts.append(data[data.files[0]])

                events = concatenate_events(parts)
                save_events_npz(npz_path, events)

        finally:
            if frame_writer is not None:
                frame_writer.close()

    elapsed = wallclock.perf_counter() - started

    if not args.quiet:
        duration_s = last_stamp / NS_PER_S
        rate = total_events / duration_s if duration_s > 0 else 0.0
        print(f"\nGenerated {total_events} events over {duration_s:.3f} s ({rate:,.0f} ev/s)")
        if total_events:
            # If you need the positive/negative split without loading the final npz,
            # track it online in the loop.
            pass
        if frame_writer is not None:
            print(f"  {n_frames} blurred frames")
        print(f"  simulated in {elapsed:.2f} s -> {args.output}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, IOError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())