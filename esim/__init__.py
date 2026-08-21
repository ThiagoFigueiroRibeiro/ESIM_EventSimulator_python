"""ESIM: an open event camera simulator.

Pure Python port of the event generation core of the original C++/ROS ESIM.
"""

from .camera_simulator import CameraSimulator, ImageBuffer
from .data_provider import FolderImageSource, StampedImage
from .event_simulator import EventSimulator
from .types import EVENT_DTYPE, EventSimConfig, empty_events

__all__ = [
    "CameraSimulator",
    "ImageBuffer",
    "FolderImageSource",
    "StampedImage",
    "EventSimulator",
    "EventSimConfig",
    "EVENT_DTYPE",
    "empty_events",
]

__version__ = "1.0.0"
