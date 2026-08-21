"""Event generation from a sequence of intensity images.

Port of ``event_camera_simulator/esim/src/event_simulator.cpp`` from the original
C++ implementation. The model is unchanged; only the loop structure differs, see
:meth:`EventSimulator.image_callback`.
"""

from typing import Optional

import numpy as np

from .types import (
    EVENT_DTYPE,
    MIN_CONTRAST_THRESHOLD,
    TOLERANCE,
    EventSimConfig,
    empty_events,
)


class EventSimulator:
    """Simulates an ideal event camera driven by a sequence of stamped images.

    The input images are assumed to be sampled at a high enough frame rate that
    per-pixel intensity can be linearly interpolated between two consecutive
    samples. Pixel reference levels are initialised from the first image, which
    therefore produces no events.

    Example:
        >>> cfg = EventSimConfig(Cp=0.2, Cm=0.2, sigma_Cp=0, sigma_Cm=0)
        >>> sim = EventSimulator(cfg)
        >>> _ = sim.image_callback(np.full((4, 4), 0.5, np.float32), 0)
        >>> events = sim.image_callback(np.full((4, 4), 0.9, np.float32), 10_000_000)
    """

    def __init__(
        self,
        config: Optional[EventSimConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config if config is not None else EventSimConfig()
        self._rng = rng if rng is not None else np.random.default_rng(self.config.random_seed)

        self._initialized = False
        self._current_time = 0
        self._ref_values: Optional[np.ndarray] = None
        self._last_img: Optional[np.ndarray] = None
        self._last_event_ts: Optional[np.ndarray] = None
        self._shape: Optional[tuple] = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def shape(self) -> Optional[tuple]:
        """Sensor size as ``(height, width)``, or ``None`` before initialisation."""
        return self._shape

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Convert an intensity image to the signal the thresholds act on."""
        img = np.asarray(img, dtype=np.float64)
        if img.ndim != 2:
            raise ValueError(f"expected a 2D grayscale image, got shape {img.shape}")
        if self.config.use_log_image:
            return np.log(self.config.log_eps + img)
        return img.copy()

    def init(self, img: np.ndarray, time: int) -> None:
        """Reset the internal state from an already preprocessed image."""
        self._initialized = True
        self._last_img = img.copy()
        self._ref_values = img.copy()
        # The C++ version stores this timestamp in seconds as a double and converts
        # back to nanoseconds on every comparison; we keep integer nanoseconds
        # throughout. A value of 0 means "this pixel has never fired", which is the
        # same sentinel the original uses.
        self._last_event_ts = np.zeros(img.shape, dtype=np.int64)
        self._current_time = int(time)
        self._shape = img.shape

    def image_callback(self, img: np.ndarray, time: int) -> np.ndarray:
        """Generate the events triggered between the previous image and this one.

        Args:
            img: Grayscale intensity image, typically in ``[0, 1]``.
            time: Absolute timestamp of ``img``, in nanoseconds.

        Returns:
            A structured array with dtype :data:`~esim.types.EVENT_DTYPE`, sorted by
            timestamp. Empty for the first image, which only initialises the state.

        The original iterates over pixels and, for each one, walks its threshold
        crossings in an inner do-while loop. Here the two loops are swapped: the outer
        loop runs over the crossing index ``k`` and every pixel is handled at once with
        NumPy. This is equivalent because within a single frame the crossings of a given
        pixel are monotonically increasing in time, so visiting them in order of ``k``
        preserves the sequential semantics that the refractory period depends on.
        """
        time = int(time)
        if time < 0:
            raise ValueError("timestamps must be non-negative")

        img = self._preprocess(img)

        if not self._initialized:
            self.init(img, time)
            return empty_events()

        if img.shape != self._shape:
            raise ValueError(f"image shape {img.shape} differs from sensor size {self._shape}")

        delta_t_ns = time - self._current_time
        if delta_t_ns <= 0:
            raise ValueError("image timestamps must be strictly increasing")

        it = self._last_img
        itdt = img
        prev_cross = self._ref_values.copy()

        diff = itdt - it
        changed = np.abs(diff) > TOLERANCE

        if not changed.any():
            self._current_time = time
            self._last_img = itdt
            return empty_events()

        pol = np.where(itdt >= it, 1.0, -1.0)
        C = np.where(pol > 0, self.config.Cp, self.config.Cm)
        sigma = np.where(pol > 0, self.config.sigma_Cp, self.config.sigma_Cm)

        # The threshold noise is drawn once per pixel per frame, not once per crossing,
        # matching the original. The clamp only applies where noise was actually added.
        if self.config.sigma_Cp > 0 or self.config.sigma_Cm > 0:
            noisy = C + self._rng.normal(0.0, 1.0, size=self._shape) * sigma
            C = np.where(sigma > 0, np.maximum(MIN_CONTRAST_THRESHOLD, noisy), C)

        n_crossings = self._count_crossings(it, itdt, prev_cross, pol, C, changed)
        n_max = int(n_crossings.max())
        if n_max == 0:
            self._current_time = time
            self._last_img = itdt
            return empty_events()

        # Guard the interpolation against a division by zero on unchanged pixels;
        # those are masked out by `changed` anyway.
        safe_diff = np.where(changed, diff, 1.0)

        refractory_ns = self.config.refractory_period_ns
        chunks = []

        for k in range(1, n_max + 1):
            active = n_crossings >= k
            curr_cross = prev_cross + pol * C * k

            # edt = (curr_cross - it) * dt / (itdt - it), truncated to whole nanoseconds
            # exactly as the C++ conversion to an integer Duration does.
            t_ev = self._current_time + (
                (curr_cross - it) * float(delta_t_ns) / safe_diff
            ).astype(np.int64)

            if refractory_ns > 0:
                never_fired = self._last_event_ts == 0
                elapsed = t_ev - self._last_event_ts
                emit = active & (never_fired | (elapsed >= refractory_ns))
            else:
                emit = active

            ys, xs = np.nonzero(emit)
            if ys.size:
                chunk = np.empty(ys.size, dtype=EVENT_DTYPE)
                chunk["x"] = xs.astype(np.uint16)
                chunk["y"] = ys.astype(np.uint16)
                chunk["t"] = t_ev[emit]
                chunk["pol"] = pol[emit] > 0
                chunks.append(chunk)
                self._last_event_ts[emit] = t_ev[emit]

            # The reference level advances on every crossing, including the ones whose
            # event was suppressed by the refractory period.
            self._ref_values[active] = curr_cross[active]

        self._current_time = time
        self._last_img = itdt

        if not chunks:
            return empty_events()

        events = np.concatenate(chunks)
        # Most event processing algorithms expect chronological order.
        return events[np.argsort(events["t"], kind="stable")]

    @staticmethod
    def _count_crossings(
        it: np.ndarray,
        itdt: np.ndarray,
        prev_cross: np.ndarray,
        pol: np.ndarray,
        C: np.ndarray,
        changed: np.ndarray,
    ) -> np.ndarray:
        """Number of threshold crossings each pixel undergoes over this interval.

        The C++ inner loop stops at the *first* ``k`` whose crossing condition fails,
        so the valid crossings are the maximal run starting at ``k = 1`` rather than
        every ``k`` that satisfies the condition. Writing the condition as

            ``prev_cross + pol * C * k`` strictly past ``it`` and not past ``itdt``

        gives ``k <= pol * (itdt - prev_cross) / C``, gated on ``k = 1`` already being
        past ``it``. Hence the closed form below.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = pol * (itdt - prev_cross) / C
            first_is_past_it = pol * (it - prev_cross) / C < 1.0

        n = np.floor(reach)
        n = np.where(np.isfinite(n), n, 0.0)
        n = np.where(changed & first_is_past_it, n, 0.0)
        return np.maximum(n, 0.0).astype(np.int64)
