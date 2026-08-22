"""Writing simulation output to disk."""

import os
from typing import Iterable, List, Optional

import cv2
import numpy as np

from .types import EVENT_DTYPE, empty_events

NS_PER_S = 1_000_000_000


def save_events_npz(path: str, events: np.ndarray) -> None:
    """Save events as a compressed ``.npz`` with one array per field."""
    _ensure_parent(path)
    np.savez_compressed(
        path,
        x=events["x"],
        y=events["y"],
        t=events["t"],
        pol=events["pol"],
    )


def load_events_npz(path: str) -> np.ndarray:
    """Load events written by :func:`save_events_npz`."""
    with np.load(path) as data:
        events = np.empty(len(data["t"]), dtype=EVENT_DTYPE)
        for field in ("x", "y", "t", "pol"):
            events[field] = data[field]
    return events


def load_events_txt(path: str) -> np.ndarray:
    """Load ``t x y pol`` events from a text export.

    Timestamps in the text file are expressed in seconds and converted back to
    integer nanoseconds. Blank lines and lines beginning with ``#`` are ignored.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.split()
            if len(columns) != 4:
                raise ValueError(f"{path}:{line_number}: expected 't x y pol'")
            try:
                t_s = float(columns[0])
                x = int(columns[1])
                y = int(columns[2])
                pol = int(columns[3])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid event") from exc
            if not np.isfinite(t_s) or t_s < 0:
                raise ValueError(f"{path}:{line_number}: timestamp must be non-negative")
            if not (0 <= x <= np.iinfo(np.uint16).max) or not (
                0 <= y <= np.iinfo(np.uint16).max
            ):
                raise ValueError(f"{path}:{line_number}: coordinates are out of range")
            if pol not in (0, 1):
                raise ValueError(f"{path}:{line_number}: polarity must be 0 or 1")
            rows.append((x, y, int(round(t_s * NS_PER_S)), bool(pol)))

    events = np.asarray(rows, dtype=EVENT_DTYPE)
    if len(events) > 1 and np.any(events["t"][1:] < events["t"][:-1]):
        events = np.sort(events, order="t", kind="stable")
    return events


def save_events_txt(path: str, events: np.ndarray) -> None:
    """Save events as text, one ``t x y pol`` per line with ``t`` in seconds.

    This is the layout used by the public event camera datasets, so the output can be
    fed to existing tooling directly.
    """
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# t x y pol\n")
        for ev in events:
            handle.write(
                f"{ev['t'] / NS_PER_S:.9f} {ev['x']} {ev['y']} {1 if ev['pol'] else 0}\n"
            )


class ImageSequenceWriter:
    """Writes a numbered PNG sequence plus a matching ``images.csv``."""

    def __init__(self, folder: str, prefix: str = "frame") -> None:
        self.folder = folder
        self.prefix = prefix
        os.makedirs(folder, exist_ok=True)
        self._entries: List[str] = []
        self._count = 0

    def add(self, image: np.ndarray, timestamp: int) -> str:
        """Write one frame, clipping to ``[0, 1]`` and scaling to 8 bits."""
        name = f"{self.prefix}_{self._count:06d}.png"
        as_u8 = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0) * 255.0
        if not cv2.imwrite(os.path.join(self.folder, name), as_u8.astype(np.uint8)):
            raise IOError(f"could not write {name} to {self.folder}")
        self._entries.append(f"{int(timestamp)},{name}")
        self._count += 1
        return name

    def close(self) -> None:
        """Flush the ``images.csv`` index."""
        with open(os.path.join(self.folder, "images.csv"), "w", encoding="utf-8") as handle:
            handle.write("# timestamp_ns, image\n")
            for entry in self._entries:
                handle.write(entry + "\n")

    def __enter__(self) -> "ImageSequenceWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def concatenate_events(chunks: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate event arrays, returning a well-formed empty array if there are none."""
    non_empty = [c for c in chunks if len(c)]
    if not non_empty:
        return empty_events()
    return np.concatenate(non_empty)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
