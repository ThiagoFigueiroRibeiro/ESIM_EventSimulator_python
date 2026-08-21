"""Tests for the event generation model.

The vectorised implementation is checked two ways: against closed-form predictions
for signals whose event train can be derived by hand, and against a literal
transcription of the original C++ per-pixel loop.
"""

import math
import unittest

import numpy as np

from esim import EventSimulator, EventSimConfig

NS_PER_S = 1_000_000_000


def reference_image_callback(state, img, time, cfg, noise):
    """Literal transcription of the C++ EventSimulator::imageCallback loop.

    Deliberately written as the original is - a per-pixel loop with an inner
    do-while over threshold crossings - so it can act as an oracle for the
    vectorised version. ``noise`` is the per-pixel threshold perturbation, passed in
    so both implementations see identical draws.
    """
    preprocessed = np.log(cfg.log_eps + img) if cfg.use_log_image else img.astype(np.float64)

    if state["last_img"] is None:
        state["last_img"] = preprocessed.copy()
        state["ref_values"] = preprocessed.copy()
        state["last_ts"] = np.zeros(img.shape, dtype=np.int64)
        state["current_time"] = time
        return []

    events = []
    tolerance = 1e-6
    delta_t_ns = time - state["current_time"]
    height, width = preprocessed.shape

    for y in range(height):
        for x in range(width):
            itdt = preprocessed[y, x]
            it = state["last_img"][y, x]
            prev_cross = state["ref_values"][y, x]

            if abs(it - itdt) <= tolerance:
                continue

            pol = 1.0 if itdt >= it else -1.0
            C = cfg.Cp if pol > 0 else cfg.Cm
            sigma_C = cfg.sigma_Cp if pol > 0 else cfg.sigma_Cm
            if sigma_C > 0:
                C = max(0.01, C + noise[y, x] * sigma_C)

            curr_cross = prev_cross
            while True:
                curr_cross += pol * C
                crossed = (pol > 0 and it < curr_cross <= itdt) or (
                    pol < 0 and itdt <= curr_cross < it
                )
                if not crossed:
                    break

                edt = int((curr_cross - it) * delta_t_ns / (itdt - it))
                t = state["current_time"] + edt
                last_stamp = state["last_ts"][y, x]
                if last_stamp == 0 or (t - last_stamp) >= cfg.refractory_period_ns:
                    events.append((x, y, t, pol > 0))
                    state["last_ts"][y, x] = t
                state["ref_values"][y, x] = curr_cross

    state["current_time"] = time
    state["last_img"] = preprocessed.copy()
    events.sort(key=lambda e: e[2])
    return events


class TestEventSimulator(unittest.TestCase):
    def _ramp(self, sim, start, stop, duration_s, steps, shape=(3, 4)):
        """Feed a uniform image whose value ramps linearly, return all events."""
        collected = []
        for i in range(steps + 1):
            frac = i / steps
            value = start + (stop - start) * frac
            t = int(round(frac * duration_s * NS_PER_S))
            collected.append(sim.image_callback(np.full(shape, value), t))
        return np.concatenate(collected)

    def test_first_image_produces_no_events(self):
        sim = EventSimulator(EventSimConfig(sigma_Cp=0, sigma_Cm=0))
        events = sim.image_callback(np.full((5, 5), 0.5), 0)
        self.assertEqual(len(events), 0)
        self.assertTrue(sim.is_initialized)
        self.assertEqual(sim.shape, (5, 5))

    def test_static_scene_produces_no_events(self):
        sim = EventSimulator(EventSimConfig(sigma_Cp=0, sigma_Cm=0))
        img = np.full((6, 7), 0.4)
        sim.image_callback(img, 0)
        for k in range(1, 10):
            self.assertEqual(len(sim.image_callback(img, k * 1_000_000)), 0)

    def test_linear_ramp_event_count_and_times(self):
        # Signal rises by 1.05 over 1 s with C = 0.2, so exactly 5 crossings occur,
        # at levels k * 0.2 and therefore at times t_k = k * 0.2 / 1.05 seconds.
        cfg = EventSimConfig(Cp=0.2, Cm=0.2, sigma_Cp=0, sigma_Cm=0, use_log_image=False)
        sim = EventSimulator(cfg)
        shape = (3, 4)
        events = self._ramp(sim, 0.0, 1.05, 1.0, steps=200, shape=shape)

        n_pixels = shape[0] * shape[1]
        self.assertEqual(len(events), 5 * n_pixels)
        self.assertTrue(np.all(events["pol"]))

        # For one pixel, check the whole event train.
        pixel = events[(events["x"] == 0) & (events["y"] == 0)]
        self.assertEqual(len(pixel), 5)
        for k, ev in enumerate(pixel, start=1):
            expected_s = k * 0.2 / 1.05
            self.assertAlmostEqual(ev["t"] / NS_PER_S, expected_s, places=4)

    def test_descending_ramp_is_symmetric(self):
        cfg = EventSimConfig(Cp=0.2, Cm=0.2, sigma_Cp=0, sigma_Cm=0, use_log_image=False)
        sim = EventSimulator(cfg)
        shape = (3, 4)
        events = self._ramp(sim, 1.05, 0.0, 1.0, steps=200, shape=shape)

        self.assertEqual(len(events), 5 * shape[0] * shape[1])
        self.assertFalse(np.any(events["pol"]))

    def test_log_image_threshold_count(self):
        # With the log signal, the number of events follows from the log ratio.
        cfg = EventSimConfig(Cp=0.3, Cm=0.3, sigma_Cp=0, sigma_Cm=0, use_log_image=True)
        sim = EventSimulator(cfg)
        start, stop = 0.05, 0.9
        expected = math.floor((math.log(cfg.log_eps + stop) - math.log(cfg.log_eps + start)) / 0.3)

        events = self._ramp(sim, start, stop, 1.0, steps=300, shape=(2, 2))
        self.assertEqual(len(events), expected * 4)

    def test_multiple_crossings_in_a_single_step(self):
        # A single large jump must emit every crossing it spans, with interpolated times.
        cfg = EventSimConfig(Cp=0.2, Cm=0.2, sigma_Cp=0, sigma_Cm=0, use_log_image=False)
        sim = EventSimulator(cfg)
        sim.image_callback(np.zeros((1, 1)), 0)
        events = sim.image_callback(np.full((1, 1), 1.05), NS_PER_S)

        self.assertEqual(len(events), 5)
        for k, ev in enumerate(events, start=1):
            self.assertAlmostEqual(ev["t"] / NS_PER_S, k * 0.2 / 1.05, places=6)

    def test_refractory_period_drops_events_but_advances_reference(self):
        # Crossings land every 0.2 s; a 0.35 s refractory period keeps every other one.
        cfg = EventSimConfig(
            Cp=0.2,
            Cm=0.2,
            sigma_Cp=0,
            sigma_Cm=0,
            use_log_image=False,
            refractory_period_ns=int(0.35 * NS_PER_S),
        )
        sim = EventSimulator(cfg)
        events = self._ramp(sim, 0.0, 1.05, 1.0, steps=200, shape=(1, 1))

        times = [ev["t"] / NS_PER_S for ev in events]
        self.assertEqual(len(times), 3)
        for actual, expected_k in zip(times, (1, 3, 5)):
            self.assertAlmostEqual(actual, expected_k * 0.2 / 1.05, places=4)

        # All five crossings happened, so the reference level advanced five times even
        # though two events were suppressed.
        self.assertAlmostEqual(float(sim._ref_values[0, 0]), 5 * 0.2, places=6)

    def test_matches_reference_implementation_without_noise(self):
        self._compare_with_reference(sigma=0.0, refractory_ns=0, seed=7)

    def test_matches_reference_implementation_with_noise(self):
        self._compare_with_reference(sigma=0.03, refractory_ns=0, seed=11)

    def test_matches_reference_implementation_with_refractory_period(self):
        self._compare_with_reference(sigma=0.0, refractory_ns=3_000_000, seed=3)

    def _compare_with_reference(self, sigma, refractory_ns, seed):
        """Run both implementations over the same random sequence and compare."""
        cfg = EventSimConfig(
            Cp=0.25,
            Cm=0.18,
            sigma_Cp=sigma,
            sigma_Cm=sigma,
            refractory_period_ns=refractory_ns,
            use_log_image=True,
        )
        shape = (6, 8)
        rng = np.random.default_rng(seed)
        frames = [rng.uniform(0.05, 0.95, size=shape) for _ in range(12)]
        times = [i * 5_000_000 for i in range(len(frames))]

        # Both implementations must observe the same threshold noise draws, so the
        # draws are generated up front and injected into each.
        noise_rng = np.random.default_rng(seed + 1000)
        draws = [noise_rng.normal(0.0, 1.0, size=shape) for _ in frames]

        class ScriptedRng:
            def __init__(self, sequence):
                self._sequence = list(sequence)

            def normal(self, loc, scale, size):
                return self._sequence.pop(0)

        # The simulator only draws when at least one sigma is positive.
        sim = EventSimulator(cfg, rng=ScriptedRng(draws[1:]) if sigma > 0 else None)

        state = {"last_img": None, "ref_values": None, "last_ts": None, "current_time": 0}
        for index, (frame, t) in enumerate(zip(frames, times)):
            actual = sim.image_callback(frame, t)
            expected = reference_image_callback(state, frame, t, cfg, draws[index])

            self.assertEqual(
                len(actual), len(expected), f"event count differs on frame {index}"
            )
            for got, want in zip(actual, expected):
                self.assertEqual(
                    (int(got["x"]), int(got["y"]), int(got["t"]), bool(got["pol"])),
                    want,
                    f"event mismatch on frame {index}",
                )

    def test_matches_reference_on_slowly_varying_signal_with_noise(self):
        """A regime the random-image comparisons do not reach.

        When the per-frame intensity step is much smaller than the threshold noise, a
        pixel's reference level can strand: no draw of C lands inside the narrow
        interval between two samples, so the pixel stops firing until the signal turns
        back past its reference. That costs a large fraction of the events, and it is
        behaviour inherited from the original model rather than an artefact of the
        vectorisation, so both implementations must agree exactly.
        """
        cfg = EventSimConfig(Cp=0.2, Cm=0.2, sigma_Cp=0.021, sigma_Cm=0.021)
        n_frames = 200
        fps = 1000.0
        # Slow sinusoid, quantised to 8 bits so many consecutive frames are identical.
        series = []
        for i in range(n_frames):
            value = 0.5 + 0.3 * math.sin(2.0 * math.pi * 5.0 * i / fps)
            series.append(round(value * 255.0) / 255.0)

        draws = [np.random.default_rng(9000 + k).normal(0.0, 1.0, (1, 1)) for k in range(n_frames)]

        class LockstepRng:
            def __init__(self):
                self.index = 0

            def normal(self, loc, scale, size):
                draw = draws[self.index]
                self.index += 1
                return draw

        sim = EventSimulator(cfg, rng=LockstepRng())
        mine = []
        for i, value in enumerate(series):
            events = sim.image_callback(np.full((1, 1), value), int(i / fps * 1e9))
            mine.extend(int(e["t"]) for e in events)

        # The reference consumes a draw on exactly the frames the port does: those
        # where the preprocessed image actually changed.
        state = {"last_img": None, "ref_values": None, "last_ts": None, "current_time": 0}
        expected = []
        draw_index = 0
        previous = None
        for i, value in enumerate(series):
            current = math.log(cfg.log_eps + value)
            if previous is not None and abs(current - previous) > 1e-6:
                noise = draws[draw_index]
                draw_index += 1
            else:
                noise = np.zeros((1, 1))
            got = reference_image_callback(
                state, np.full((1, 1), value), int(i / fps * 1e9), cfg, noise
            )
            expected.extend(e[2] for e in got)
            previous = current

        self.assertEqual(mine, expected)
        # Noise really does starve this signal: without it the same input fires more.
        quiet = EventSimulator(EventSimConfig(Cp=0.2, Cm=0.2, sigma_Cp=0, sigma_Cm=0))
        quiet_count = sum(
            len(quiet.image_callback(np.full((1, 1), v), int(i / fps * 1e9)))
            for i, v in enumerate(series)
        )
        self.assertGreater(quiet_count, len(mine))

    def test_events_are_sorted_by_timestamp(self):
        cfg = EventSimConfig(Cp=0.15, Cm=0.15, sigma_Cp=0, sigma_Cm=0)
        sim = EventSimulator(cfg)
        rng = np.random.default_rng(0)
        sim.image_callback(rng.uniform(0.1, 0.9, (10, 10)), 0)
        for i in range(1, 6):
            events = sim.image_callback(rng.uniform(0.1, 0.9, (10, 10)), i * 2_000_000)
            self.assertTrue(np.all(np.diff(events["t"]) >= 0))

    def test_seeded_noise_is_reproducible(self):
        rng_frames = np.random.default_rng(5)
        frames = [rng_frames.uniform(0.1, 0.9, (8, 8)) for _ in range(4)]

        def run():
            sim = EventSimulator(EventSimConfig(Cp=0.2, Cm=0.2, random_seed=42))
            return np.concatenate(
                [sim.image_callback(f, i * 1_000_000) for i, f in enumerate(frames)]
            )

        np.testing.assert_array_equal(run(), run())

    def test_rejects_invalid_input(self):
        sim = EventSimulator(EventSimConfig(sigma_Cp=0, sigma_Cm=0))
        sim.image_callback(np.full((4, 4), 0.5), 1_000)

        with self.assertRaises(ValueError):
            sim.image_callback(np.full((5, 5), 0.5), 2_000)
        with self.assertRaises(ValueError):
            sim.image_callback(np.full((4, 4), 0.5), 500)
        with self.assertRaises(ValueError):
            sim.image_callback(np.zeros((4, 4, 3)), 3_000)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            EventSimConfig(Cp=0)
        with self.assertRaises(ValueError):
            EventSimConfig(sigma_Cm=-1)
        with self.assertRaises(ValueError):
            EventSimConfig(refractory_period_ns=-5)


if __name__ == "__main__":
    unittest.main()
