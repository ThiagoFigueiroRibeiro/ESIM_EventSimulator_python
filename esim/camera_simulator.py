"""Conventional camera simulation, including motion blur.

Port of ``event_camera_simulator/esim/src/camera_simulator.cpp`` and its header.
"""

from collections import deque
from typing import Deque, NamedTuple, Optional, Tuple

import numpy as np

NS_PER_MS = 1_000_000


class _ImageData(NamedTuple):
    image: np.ndarray
    stamp: int
    #: Time elapsed since the previous image, 0 for the first one.
    exposure_time: int


class ImageBuffer:
    """Rolling buffer holding the images that fall within one exposure window."""

    def __init__(self, buffer_size_ns: int) -> None:
        if buffer_size_ns <= 0:
            raise ValueError("buffer size must be strictly positive")
        self._buffer_size_ns = int(buffer_size_ns)
        self._data: Deque[_ImageData] = deque()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def exposure_time_ns(self) -> int:
        return self._buffer_size_ns

    @property
    def data(self) -> Tuple[_ImageData, ...]:
        return tuple(self._data)

    def add_image(self, t: int, img: np.ndarray) -> None:
        """Append an image and drop everything older than one exposure window."""
        t = int(t)
        if self._data and t <= self._data[-1].stamp:
            raise ValueError("image timestamps must be strictly increasing")

        exposure_time = 0 if not self._data else t - self._data[-1].stamp
        self._data.append(_ImageData(np.array(img, dtype=np.float64), t, exposure_time))

        oldest_allowed = t - self._buffer_size_ns
        while self._data and self._data[0].stamp < oldest_allowed:
            self._data.popleft()


class CameraSimulator:
    """Turns a high-rate sequence of irradiance images into blurred camera frames.

    Each input image is treated as an instantaneous measure of irradiance. A frame is
    produced by averaging every image inside the exposure window, weighted by how long
    that image was the current one.
    """

    def __init__(self, exposure_time_ms: float) -> None:
        if exposure_time_ms <= 0:
            raise ValueError("exposure time must be strictly positive")
        self.exposure_time_ns = int(round(exposure_time_ms * NS_PER_MS))
        self._buffer = ImageBuffer(self.exposure_time_ns)
        self._initial_time: Optional[int] = None

    def image_callback(self, img: np.ndarray, time: int) -> Optional[np.ndarray]:
        """Add an image to the exposure window and return the blurred frame.

        Args:
            img: Irradiance image.
            time: Absolute timestamp of ``img``, in nanoseconds.

        Returns:
            The blurred frame, or ``None`` while the buffered images do not yet span a
            full exposure time. The frame corresponds to mid-exposure, i.e.
            ``time - exposure_time_ns / 2``.
        """
        time = int(time)
        self._buffer.add_image(time, img)

        # The C++ version keeps this reference time in a function-local `static`, which
        # makes it shared by every CameraSimulator in the process; here it is per
        # instance, so a multi-camera rig behaves correctly.
        if self._initial_time is None:
            self._initial_time = time

        if time - self._initial_time < self.exposure_time_ns:
            return None

        # The original converts each weight to whole milliseconds before accumulating,
        # which truncates any interval shorter than 1 ms to a weight of zero. Since the
        # weights are normalised, using the raw nanosecond durations is the same
        # computation without that failure mode at high frame rates.
        weights = np.array([d.exposure_time for d in self._buffer.data], dtype=np.float64)
        denom = weights.sum()
        if denom <= 0:
            return None

        stack = np.stack([d.image for d in self._buffer.data])
        return np.tensordot(weights, stack, axes=(0, 0)) / denom

    def mid_exposure_time(self, time: int) -> int:
        """Timestamp a frame captured at ``time`` should carry."""
        return int(time) - self.exposure_time_ns // 2
