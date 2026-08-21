"""Generate the images.csv index that esim reads from an image folder.

Assigns evenly spaced timestamps to the images found in a folder, in sorted order.

    python tools/generate_stamps_file.py -i /data/frames -r 1000
"""

import argparse
import os

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input_folder", required=True, help="folder containing the images")
    parser.add_argument(
        "-r", "--framerate", default=1000.0, type=float, help="video framerate, in Hz"
    )
    args = parser.parse_args(argv)

    if args.framerate <= 0:
        parser.error("framerate must be strictly positive")

    images = sorted(
        f for f in os.listdir(args.input_folder) if f.lower().endswith(IMAGE_SUFFIXES)
    )
    if not images:
        parser.error(f"no images found in {args.input_folder}")

    csv_path = os.path.join(args.input_folder, "images.csv")
    dt_ns = int(round(1e9 / args.framerate))

    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("# timestamp_ns, image\n")
        for index, name in enumerate(images):
            handle.write(f"{index * dt_ns},{name}\n")

    print(f"Wrote {csv_path}: {len(images)} images at {args.framerate} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
