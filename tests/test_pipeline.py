"""End-to-end tests: folder input, CLI, output files and visualisation helpers."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from esim import FolderImageSource
from esim.cli import main as cli_main
from esim.viz import accumulate, to_rgb
from esim.writers import load_events_npz, save_events_npz, save_events_txt
from esim.types import EVENT_DTYPE

import tools.make_test_sequence as make_test_sequence


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="esim_test_")
        cls.seq = os.path.join(cls.tmp, "seq")
        make_test_sequence.main(
            ["--output", cls.seq, "--frames", "60", "--width", "48", "--height", "36"]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_folder_source_reads_stamps_and_images(self):
        source = FolderImageSource(self.seq)
        self.assertEqual(len(source), 60)

        stamps = []
        for stamp, image in source:
            stamps.append(stamp)
            self.assertEqual(image.shape, (36, 48))
            self.assertGreaterEqual(image.min(), 0.0)
            self.assertLessEqual(image.max(), 1.0)

        self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])))

    def test_folder_source_rejects_missing_index(self):
        with self.assertRaises(FileNotFoundError):
            FolderImageSource(self.tmp)

    def test_folder_source_rejects_malformed_line(self):
        broken = os.path.join(self.tmp, "broken")
        os.makedirs(broken, exist_ok=True)
        with open(os.path.join(broken, "images.csv"), "w", encoding="utf-8") as handle:
            handle.write("# comment\nnot-a-number,frame.png\n")
        with self.assertRaises(ValueError):
            FolderImageSource(broken)

    def test_cli_end_to_end(self):
        out = os.path.join(self.tmp, "out")
        code = cli_main(
            [
                "--input", self.seq,
                "--output", out,
                "--contrast-threshold", "0.2",
                "--contrast-threshold-sigma-pos", "0",
                "--contrast-threshold-sigma-neg", "0",
                "--exposure-time-ms", "5",
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)

        self.assertTrue(os.path.isfile(os.path.join(out, "events.npz")))
        self.assertTrue(os.path.isfile(os.path.join(out, "events.txt")))
        self.assertTrue(os.path.isfile(os.path.join(out, "frames", "images.csv")))

        events = load_events_npz(os.path.join(out, "events.npz"))
        self.assertGreater(len(events), 0)
        self.assertTrue(np.all(np.diff(events["t"]) >= 0))
        self.assertLess(events["x"].max(), 48)
        self.assertLess(events["y"].max(), 36)

        # A translating grating fires both polarities in comparable numbers.
        positive = int(np.count_nonzero(events["pol"]))
        ratio = positive / len(events)
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 0.7)

        # The blurred frames are readable and correctly indexed.
        frames = FolderImageSource(os.path.join(out, "frames"))
        self.assertGreater(len(frames), 0)

    def test_cli_reports_missing_input(self):
        code = cli_main(["--input", os.path.join(self.tmp, "nope"), "--output", self.tmp, "--quiet"])
        self.assertEqual(code, 1)

    def test_cli_argfile(self):
        argfile = os.path.join(self.tmp, "run.conf")
        out = os.path.join(self.tmp, "out_argfile")
        with open(argfile, "w", encoding="utf-8") as handle:
            handle.write(f"--input\n{self.seq}\n--output\n{out}\n--quiet\n--no-txt\n")

        self.assertEqual(cli_main(["@" + argfile]), 0)
        self.assertTrue(os.path.isfile(os.path.join(out, "events.npz")))
        self.assertFalse(os.path.isfile(os.path.join(out, "events.txt")))

    def test_event_roundtrip_npz_and_txt(self):
        events = np.empty(3, dtype=EVENT_DTYPE)
        events["x"] = [1, 2, 3]
        events["y"] = [4, 5, 6]
        events["t"] = [10, 2_000_000_000, 3_500_000_000]
        events["pol"] = [True, False, True]

        npz_path = os.path.join(self.tmp, "roundtrip.npz")
        save_events_npz(npz_path, events)
        np.testing.assert_array_equal(load_events_npz(npz_path), events)

        txt_path = os.path.join(self.tmp, "roundtrip.txt")
        save_events_txt(txt_path, events)
        with open(txt_path, encoding="utf-8") as handle:
            lines = [line for line in handle if not line.startswith("#")]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1].split(), ["2.000000000", "2", "5", "0"])

    def test_accumulate_and_render(self):
        events = np.empty(4, dtype=EVENT_DTYPE)
        events["x"] = [0, 0, 1, 1]
        events["y"] = [0, 0, 1, 1]
        events["t"] = [1, 2, 3, 4]
        events["pol"] = [True, True, False, False]

        frame = accumulate(events, shape=(2, 2))
        self.assertEqual(frame[0, 0], 2)
        self.assertEqual(frame[1, 1], -2)

        rgb = to_rgb(frame)
        self.assertEqual(rgb.shape, (2, 2, 3))
        # ON pixels lean blue, OFF pixels lean red.
        self.assertGreater(rgb[0, 0, 2], rgb[0, 0, 0])
        self.assertGreater(rgb[1, 1, 0], rgb[1, 1, 2])

    def test_accumulate_time_window(self):
        events = np.empty(4, dtype=EVENT_DTYPE)
        events["x"] = [0, 0, 0, 0]
        events["y"] = [0, 0, 0, 0]
        events["t"] = [10, 20, 30, 40]
        events["pol"] = [True, True, True, True]

        self.assertEqual(accumulate(events, shape=(1, 1), t_start=20, t_end=40)[0, 0], 2)


if __name__ == "__main__":
    unittest.main()
