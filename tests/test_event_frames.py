import os
import tempfile
import unittest

import cv2
import numpy as np

from esim.event_frames import infer_shape, write_event_frames
from esim.types import EVENT_DTYPE
from esim.writers import load_events_txt


class TestEventFrames(unittest.TestCase):
    def test_load_txt_and_render_time_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "events.txt")
            output = os.path.join(tmp, "frames")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("# t x y pol\n")
                handle.write("0.001 1 0 1\n")
                handle.write("0.012 0 1 0\n")

            events = load_events_txt(source)
            self.assertEqual(events.dtype, EVENT_DTYPE)
            self.assertEqual(infer_shape(events, None, None), (2, 2))
            self.assertEqual(write_event_frames(events, output, 10_000_000, (2, 2)), 2)

            first = cv2.imread(os.path.join(output, "frame_000000.png"))
            second = cv2.imread(os.path.join(output, "frame_000001.png"))
            self.assertEqual(first[0, 1].tolist(), [255, 0, 0])  # ON is blue (BGR)
            self.assertEqual(second[1, 0].tolist(), [0, 0, 255])  # OFF is red (BGR)
            self.assertTrue(os.path.isfile(os.path.join(output, "images.csv")))

    def test_rejects_sensor_shape_smaller_than_coordinates(self):
        events = np.array([(4, 3, 0, True)], dtype=EVENT_DTYPE)
        with self.assertRaisesRegex(ValueError, "smaller than event coordinates"):
            infer_shape(events, width=4, height=4)


if __name__ == "__main__":
    unittest.main()
