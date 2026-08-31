# ESIM (Python port): an event camera simulator

A pure-Python port of the event-generation core of [ESIM](https://github.com/uzh-rpg/rpg_esim), an open-source simulator for event cameras (DVS/DAVIS-class sensors). Given a folder of timestamped intensity images, it reproduces the original per-pixel event model — including threshold noise, the refractory period, and motion-blurred frame output — without any ROS, catkin, or C++ toolchain.

```bibtex
@Article{Rebecq18corl,
  author        = {Henri Rebecq and Daniel Gehrig and Davide Scaramuzza},
  title         = {{ESIM}: an Open Event Camera Simulator},
  journal       = {Conf. on Robotics Learning (CoRL)},
  year          = 2018,
  month         = oct
}
```

The paper is available [here](http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf). If you use this code, please cite the publication above.

## What this is (and isn't)

This repository ports only the **event generation pipeline** of the original C++/ROS project — the part that turns a sequence of intensity images into events. It does **not** include the original's scene renderers (planar, panorama, OpenGL, UnrealCV), trajectory/IMU simulation, or ROS publishing/rosbag recording. If you need those, use the [original C++/ROS ESIM](https://github.com/uzh-rpg/rpg_esim) or the [GPU-accelerated Python bindings](https://github.com/uzh-rpg/rpg_vid2e), which wrap the same reference implementation.

In practice this means: **you supply the images** (rendered however you like, or a real video/photo sequence), and this tool simulates what an event camera would have seen.

## Features

- Faithful port of the C++ event model: log- or linear-intensity thresholding, separate positive/negative contrast thresholds (C+/C-), additive Gaussian noise on the thresholds, and a per-pixel refractory period
- Motion-blurred frame synthesis via a finite exposure time, alongside the event stream
- Simple folder-based input (`images.csv` + image files) and file-based output (`.npz` / `.txt` events, PNG frame sequence)
- A small, dependency-light Python API (`esim.EventSimulator`, `esim.CameraSimulator`, `esim.EventSimConfig`) usable outside the CLI
- Visualization helpers for accumulated event frames and event-rate plots (`esim.viz`)
- Event-frame export from an event stream (`esim.event_frames`) for visualization or downstream processing
- A webcam-based pseudo-event demo (`esim.event_frames_from_camera`) that approximates event-camera output in real time
- Only NumPy, OpenCV, and Matplotlib as dependencies — no ROS, no compiled extensions, runs anywhere Python does (Windows, macOS, Linux)

## Architecture

```text
images.csv + frames ──▶ FolderImageSource ──▶ EventSimulator ──▶ events.npz / events.txt
                                          ├──▶ CameraSimulator ──▶ frames/ (blurred PNGs)
                                          ├──▶ Viz helpers ──▶ event summary plots
                                          └──▶ Event-frame exporter ──▶ PNG sequence
```

- **`FolderImageSource`** ([esim/data_provider.py](esim/data_provider.py)) reads a stamped image sequence from disk.
- **`EventSimulator`** ([esim/event_simulator.py](esim/event_simulator.py)) compares the (log-)intensity signal against the contrast thresholds per pixel and emits events, honoring threshold noise and the refractory period.
- **`CameraSimulator`** ([esim/camera_simulator.py](esim/camera_simulator.py)) integrates intensity over an exposure window to synthesize motion-blurred conventional frames.
- **`esim.cli`** ([esim/cli.py](esim/cli.py)) wires the image sequence input together with the event/camera simulators into the `python -m esim.cli` command-line tool.
- **`esim.event_frames`** ([esim/event_frames.py](esim/event_frames.py)) converts an event stream into a timestamped PNG event-frame sequence.
- **`esim.event_frames_from_camera`** ([esim/event_frames_from_camera.py](esim/event_frames_from_camera.py)) reconstructs pseudo-events from a webcam stream in real time for demo purposes.
- **`esim.writers`** ([esim/writers.py](esim/writers.py)) and **`esim.viz`** ([esim/viz.py](esim/viz.py)) handle output I/O and visualization.

## Repository layout

| Path | Contents |
| --- | --- |
| [`esim/`](esim) | The simulator package: types, event/camera simulators, path-based image input, CLI, writers, event-frame generation, and visualization |
| [`tests/`](tests) | Unit and end-to-end tests (`unittest` / `pytest`) |
| [`tools/`](tools) | Standalone scripts: synthetic test-sequence generator, `images.csv` builder, video frame extractor, event-center analysis, and PCA tracking utilities |
| [`requirements.txt`](requirements.txt) | Runtime dependencies |
| [`doc/`](doc) | Additional notes and walkthroughs, including video conversion examples |

## Requirements

- Python 3.8+
- `numpy`, `opencv-python`, `matplotlib` (see [requirements.txt](requirements.txt))
- Runs on Windows, macOS, and Linux — no ROS, catkin, vcstool, or C++ build tools needed
- Works inside a plain `venv` or a conda environment; nothing here requires conda specifically

## Installation

```bash
conda create -n esim python=3.10
conda activate esim
pip install -r requirements.txt
```

(A regular `venv` works identically — swap the first two lines for `python -m venv .venv` and activating it.)

## Preparing input

The simulator reads a folder containing an `images.csv` index and the image files it references:

```text
seq/
├── images.csv
├── frame_000000.png
├── frame_000001.png
└── ...
```

`images.csv` has one `timestamp_ns,filename` pair per line (lines starting with `#` or `%` are comments):

```text
# timestamp_ns, image
0,frame_000000.png
1000000,frame_000001.png
```

Several helper scripts are provided:

- **`tools/generate_stamps_file.py`** builds `images.csv` for a folder of images you already have, at a fixed frame rate:
  ```bash
  python tools/generate_stamps_file.py -i path/to/frames -r 1000
  ```
- **`tools/make_test_sequence.py`** renders a synthetic translating grating end to end — useful for a quick demo or for tests, since it produces a dense, predictable event stream:
  ```bash
  python tools/make_test_sequence.py --output demo_seq --frames 200
  ```
- **`tools/prepare_video.py`** extracts frames from a video file (`.mp4` and other OpenCV-decodable formats) and writes the matching `images.csv`:
  ```bash
  python tools/prepare_video.py -i video/video.mp4 -o video_input
  ```
- **`tools/calculate_centers.py`** groups events into time bins and plots the per-bin barycenter (center of mass) for each frame sequence:
  ```bash
  python tools/calculate_centers.py --npz video_out/events.npz --fps 60 --first-events 50 --baricenter-frames ./baricenter_frames
  ```
- **`tools/calculate_centers_PCA.py`** performs a PCA-based center estimate and overlays the principal axes on the event cloud for each time bin:
  ```bash
  python tools/calculate_centers_PCA.py --npz video_out/events.npz --fps 60 --first-events 50 --PCA-frames ./pca_frames
  ```
  See [doc/converter_video.md](doc/converter_video.md) (in Portuguese) for the full video-to-event-frames walkthrough.

## Running the simulator

```bash
python -m esim.cli --input video_seq --output video_out --contrast-threshold 0.2
```

(The default contrast threshold of `1.0` is tuned for full-range renders; the synthetic demo grating above has a modest contrast, so a lower threshold like `0.2` is needed to actually trigger events. Tune it to match your own image sequence's contrast.)

Arguments can also be kept in a file and loaded with `@`, one flag per line (this mirrors the flagfiles the original C++ tool used):

```bash
python -m esim.cli @cfg/my_run.conf
```

### Flags

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | *(required)* | Folder containing `images.csv` and the images |
| `-o`, `--output` | *(required)* | Folder to write results into |
| `--contrast-threshold` | — | Set both C+ and C- at once (overrides the two below) |
| `--contrast-threshold-pos` | `1.0` | Positive (ON) contrast threshold, C+ |
| `--contrast-threshold-neg` | `1.0` | Negative (OFF) contrast threshold, C- |
| `--contrast-threshold-sigma-pos` | `0.0` | Std. dev. of Gaussian noise added to C+ |
| `--contrast-threshold-sigma-neg` | `0.0` | Std. dev. of Gaussian noise added to C- |
| `--refractory-period-ns` | `0` | Minimum time between two events at the same pixel |
| `--no-log-image` | off | Threshold raw intensity instead of log intensity |
| `--log-eps` | `0.001` | Epsilon added before the log, to stabilize dark pixels |
| `--random-seed` | — | Seed for the threshold noise (nondeterministic if unset) |
| `--exposure-time-ms` | `10.0` | Exposure time used to synthesize motion blur |
| `--no-blurred-frames` | off | Skip motion-blurred frame output entirely |
| `--no-txt` | off | Skip the `events.txt` export (still writes `events.npz`) |
| `--quiet` | off | Suppress progress output |

`--contrast-threshold-sigma-pos/neg` default to `0` here rather than the original's `0.021`, matching every configuration the original ESIM ships with: threshold noise starves the event stream when the per-frame intensity step is much smaller than the noise itself.

### Output

```text
video_out/
├── events.npz          # x, y, t (ns), pol — see esim.writers.load_events_npz
├── events.txt          # "t x y pol" per line, t in seconds (omit with --no-txt)
└── frames/             # blurred frames + images.csv (omit with --no-blurred-frames)
```

## Live pseudo-event demo

A lightweight webcam demo is included for inspecting event-like behavior in real time without a precomputed image sequence:

```bash
python -m esim.event_frames_from_camera --camera 0 --width 640 --height 480 --on-threshold 0.2 --off-threshold 0.2 --window-ms 50 --refractory-ms 5
```

This opens a side-by-side window showing the live camera feed on the left and a pseudo-event reconstruction on the right. It is intended as a visualization/debugging aid rather than a full event-camera simulation pipeline.

## Visualizing results

```bash
python -m esim.viz video_out/events.npz
python -m esim.viz video_out --save-to accumulated_events.png   # headless, writes a PNG instead of a window
```

This renders the accumulated event image (blue = net ON, red = net OFF) next to the event-rate-over-time curve.

![alt text](accumulated_events.png)

To convert an event stream into an event-frame sequence (green = ON, red = OFF), accumulating events in fixed windows:

```bash
python -m esim.event_frames video_out/events.npz --output video_out/event_frames --window-ms 10
```

Or, if you know the source video FPS and want the window size to be half a frame period:

```bash
python -m esim.event_frames video_out/events.npz --output video_out/event_frames --fps 25
```

This uses:

- `window_ms = 1000 / (2 * fps)`

So for `25 fps`, the window becomes `20 ms`.

This writes numbered PNGs plus an `images.csv` timestamp index. Use a shorter window for finer temporal detail or a longer one to accumulate more events per image.

## Example Video

You can see the ESIM simulator in action with an example video demonstrating event generation:

![Example gif](example.gif)

Source: https://www.youtube.com/watch?v=QfDoQwIAaXg

This video shows the transformation from RGB images to simulated event camera output.

## Using it as a library

```python
from esim import EventSimulator, EventSimConfig, CameraSimulator

sim = EventSimulator(EventSimConfig(Cp=0.2, Cm=0.2, random_seed=0))
camera = CameraSimulator(exposure_time_ms=10.0)

for stamp_ns, image in my_image_sequence:       # image: 2D array in [0, 1]
    events = sim.image_callback(image, stamp_ns)  # structured array, esim.EVENT_DTYPE
    frame = camera.image_callback(image, stamp_ns)  # None until one exposure window is filled
```

## Running the tests

```bash
python -m pytest tests/ -q
```

## Acknowledgements

This is a port of [ESIM](https://github.com/uzh-rpg/rpg_esim) by the Robotics and Perception Group (University of Zurich). All credit for the underlying event-generation model goes to the original authors; see the citation above.

## License

Released under the MIT License. See [LICENSE](LICENSE).