"""Tests for the conventional camera simulation (motion blur)."""

import unittest

import numpy as np

from esim import CameraSimulator, ImageBuffer

NS_PER_MS = 1_000_000


class TestImageBuffer(unittest.TestCase):
    def test_exposure_time_is_gap_since_previous_image(self):
        buf = ImageBuffer(10 * NS_PER_MS)
        buf.add_image(0, np.zeros((2, 2)))
        buf.add_image(3 * NS_PER_MS, np.zeros((2, 2)))
        buf.add_image(7 * NS_PER_MS, np.zeros((2, 2)))

        self.assertEqual([d.exposure_time for d in buf.data], [0, 3 * NS_PER_MS, 4 * NS_PER_MS])

    def test_drops_images_older_than_the_window(self):
        buf = ImageBuffer(10 * NS_PER_MS)
        for k in range(0, 26, 5):
            buf.add_image(k * NS_PER_MS, np.full((2, 2), float(k)))

        stamps = [d.stamp for d in buf.data]
        self.assertEqual(stamps, [15 * NS_PER_MS, 20 * NS_PER_MS, 25 * NS_PER_MS])
        span = stamps[-1] - stamps[0]
        self.assertLessEqual(span, buf.exposure_time_ns)

    def test_rejects_non_monotonic_timestamps(self):
        buf = ImageBuffer(10 * NS_PER_MS)
        buf.add_image(5 * NS_PER_MS, np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            buf.add_image(5 * NS_PER_MS, np.zeros((2, 2)))

    def test_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            ImageBuffer(0)


class TestCameraSimulator(unittest.TestCase):
    def test_returns_none_until_exposure_window_is_covered(self):
        cam = CameraSimulator(exposure_time_ms=10.0)
        for k in range(0, 10):
            self.assertIsNone(cam.image_callback(np.full((3, 3), 0.5), k * NS_PER_MS))
        self.assertIsNotNone(cam.image_callback(np.full((3, 3), 0.5), 10 * NS_PER_MS))

    def test_constant_sequence_passes_through_unchanged(self):
        cam = CameraSimulator(exposure_time_ms=5.0)
        img = np.full((4, 4), 0.37)
        frame = None
        for k in range(0, 12):
            frame = cam.image_callback(img, k * NS_PER_MS)
        np.testing.assert_allclose(frame, img)

    def test_blur_is_the_time_weighted_mean_of_the_window(self):
        # Uniform 1 ms spacing, so the frame is the plain mean of the images that are
        # still inside the 4 ms window (the oldest contributes weight 0 as its
        # exposure_time refers to the gap before it).
        cam = CameraSimulator(exposure_time_ms=4.0)
        values = [0.0, 0.2, 0.4, 0.6, 0.8]
        frame = None
        for k, v in enumerate(values):
            frame = cam.image_callback(np.full((2, 2), v), k * NS_PER_MS)

        # Window holds stamps 0..4 ms; weights are [0, 1, 1, 1, 1] ms.
        expected = np.mean(values[1:])
        np.testing.assert_allclose(frame, np.full((2, 2), expected))

    def test_sub_millisecond_frames_are_weighted_correctly(self):
        # The C++ version truncates each weight to whole milliseconds, which would make
        # every weight here zero. Using nanosecond durations keeps the mean well defined.
        cam = CameraSimulator(exposure_time_ms=1.0)
        step_ns = 100_000  # 0.1 ms
        values = [0.1 * i for i in range(11)]
        frame = None
        for k, v in enumerate(values):
            frame = cam.image_callback(np.full((2, 2), v), k * step_ns)

        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame, np.full((2, 2), np.mean(values[1:])))

    def test_mid_exposure_timestamp(self):
        cam = CameraSimulator(exposure_time_ms=10.0)
        self.assertEqual(cam.mid_exposure_time(100 * NS_PER_MS), 95 * NS_PER_MS)

    def test_rejects_invalid_exposure(self):
        with self.assertRaises(ValueError):
            CameraSimulator(exposure_time_ms=0)


if __name__ == "__main__":
    unittest.main()
