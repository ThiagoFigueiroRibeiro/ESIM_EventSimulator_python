"""Core data types and configuration for the ESIM event camera simulator."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Structured dtype used for every event array produced by the simulator.
#: ``t`` is an absolute timestamp in nanoseconds, matching the C++ implementation.
EVENT_DTYPE = np.dtype(
    [
        ("x", np.uint16),
        ("y", np.uint16),
        ("t", np.int64),
        ("pol", np.bool_),
    ]
)

#: A contrast threshold is never allowed to fall below this value once noise is added.
MIN_CONTRAST_THRESHOLD = 0.01

#: Intensity changes smaller than this are treated as no change at all.
TOLERANCE = 1e-6


def empty_events() -> np.ndarray:
    """Return an empty event array with the canonical dtype."""
    return np.empty(0, dtype=EVENT_DTYPE)


@dataclass
class EventSimConfig:
    """Parameters of the event generation model.

    Defaults mirror the gflags defaults of the original C++ implementation.

    Attributes:
        Cp: Nominal contrast threshold for positive (ON) events.
        Cm: Nominal contrast threshold for negative (OFF) events.
        sigma_Cp: Standard deviation of the Gaussian noise added to ``Cp``.
        sigma_Cm: Standard deviation of the Gaussian noise added to ``Cm``.
        refractory_period_ns: Minimum time between two events at the same pixel.
        use_log_image: Operate on log intensity rather than raw intensity.
        log_eps: Epsilon added before taking the log, to stabilise dark pixels.
        random_seed: Seed for the contrast-threshold noise. ``None`` is nondeterministic.
    """

    Cp: float = 1.0
    Cm: float = 1.0
    sigma_Cp: float = 0.021
    sigma_Cm: float = 0.021
    refractory_period_ns: int = 0
    use_log_image: bool = True
    log_eps: float = 0.001
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.Cp <= 0 or self.Cm <= 0:
            raise ValueError("contrast thresholds must be strictly positive")
        if self.sigma_Cp < 0 or self.sigma_Cm < 0:
            raise ValueError("contrast threshold sigmas must be non-negative")
        if self.refractory_period_ns < 0:
            raise ValueError("refractory period must be non-negative")
