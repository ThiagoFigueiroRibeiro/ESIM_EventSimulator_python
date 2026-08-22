"""Extract frames from a video file into an esim-ready image sequence.

Reads a video (``.mp4`` and other OpenCV-decodable formats), writes each frame as a
numbered PNG into an output folder (default: ``video_input``), and writes the
``images.csv`` timestamp index that :class:`esim.data_provider.FolderImageSource`
expects.

    python tools/premiere_video.py -i video/video.mp4
    python tools/premiere_video.py -i video/video.mp4 -o video_input --framerate 240
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NS_PER_MS = 1_000_000
SUPPORTED_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def extract_frames(input_path: str, output_folder: str, framerate: float = None):
    """Decode every frame of ``input_path`` into ``output_folder`` plus its images.csv."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"no such video file: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if framerate is not None:
        fps = framerate
    elif source_fps and source_fps > 0:
        fps = source_fps
    else:
        cap.release()
        raise ValueError("could not read the video's framerate; pass --framerate explicitly")
    dt_ns = int(round(1e9 / fps))

    os.makedirs(output_folder, exist_ok=True)

    entries = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        name = f"frame_{index:06d}.png"
        if not cv2.imwrite(os.path.join(output_folder, name), frame):
            cap.release()
            raise IOError(f"could not write {name} to {output_folder}")
        # CAP_PROP_POS_MSEC is unreliable on some backends/codecs (stuck at 0, or
        # non-monotonic); entries are patched below with fixed-framerate timestamps
        # whenever that happens.
        stamp_ns = int(round(pos_msec * NS_PER_MS)) if pos_msec and pos_msec > 0 else -1
        entries.append([stamp_ns, name])
        index += 1

    cap.release()

    if not entries:
        raise ValueError(f"no frames decoded from {input_path}")

    stamps = [s for s, _ in entries]
    unreliable = any(s < 0 for s in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:]))
    if unreliable:
        for i, entry in enumerate(entries):
            entry[0] = i * dt_ns

    csv_path = os.path.join(output_folder, "images.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("# timestamp_ns, image\n")
        for stamp_ns, name in entries:
            handle.write(f"{stamp_ns},{name}\n")

    return len(entries), fps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="path to the video file (.mp4, ...)")
    parser.add_argument(
        "-o",
        "--output",
        default="video_input",
        help="folder for the frames and images.csv (default: video_input)",
    )
    parser.add_argument(
        "--framerate",
        type=float,
        default=None,
        help="override the video's reported framerate, in Hz",
    )
    args = parser.parse_args(argv)

    suffix = os.path.splitext(args.input)[1].lower()
    if suffix not in SUPPORTED_SUFFIXES:
        parser.error(f"unsupported video format {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")

    if args.framerate is not None and args.framerate <= 0:
        parser.error("--framerate must be positive")

    try:
        count, fps = extract_frames(args.input, args.output, args.framerate)
    except (FileNotFoundError, ValueError, IOError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {count} frames ({fps:g} fps) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
