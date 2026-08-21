"""Reading a stamped image sequence from a folder.

Port of ``event_camera_simulator/esim_data_provider/src/data_provider_from_folder.cpp``.
"""

import os
from typing import Iterator, List, NamedTuple, Tuple

import cv2
import numpy as np

IMAGES_CSV = "images.csv"
_COMMENT_PREFIXES = ("#", "%")


class StampedImage(NamedTuple):
    #: Absolute timestamp, in nanoseconds.
    timestamp: int
    #: Grayscale image scaled to [0, 1].
    image: np.ndarray


class FolderImageSource:
    """Iterates over the images listed in a folder's ``images.csv``.

    The CSV holds one ``timestamp_ns,filename`` pair per line; lines starting with
    ``#`` or ``%`` are comments. This is the format produced by
    ``tools/generate_stamps_file.py``.

    Example:
        >>> for stamp, img in FolderImageSource("/data/seq"):  # doctest: +SKIP
        ...     print(stamp, img.shape)
    """

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.csv_path = os.path.join(folder, IMAGES_CSV)
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"no {IMAGES_CSV} found in {folder}")
        self._entries = self._parse(self.csv_path)
        if not self._entries:
            raise ValueError(f"{self.csv_path} lists no images")

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[Tuple[int, str]]:
        """The parsed ``(timestamp_ns, filename)`` pairs."""
        return list(self._entries)

    @staticmethod
    def _parse(csv_path: str) -> List[Tuple[int, str]]:
        entries: List[Tuple[int, str]] = []
        with open(csv_path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith(_COMMENT_PREFIXES):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    raise ValueError(f"{csv_path}:{lineno}: expected 'timestamp,filename'")
                try:
                    stamp = int(parts[0])
                except ValueError as exc:
                    raise ValueError(f"{csv_path}:{lineno}: bad timestamp {parts[0]!r}") from exc
                entries.append((stamp, parts[1]))
        return entries

    def __iter__(self) -> Iterator[StampedImage]:
        for stamp, name in self._entries:
            path = os.path.join(self.folder, name)
            raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(f"could not load image {path}")
            yield StampedImage(stamp, raw.astype(np.float64) / 255.0)
