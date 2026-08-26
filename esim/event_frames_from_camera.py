"""
Real-time pseudo-event reconstruction from a webcam stream.

Left:  camera image
Right: reconstructed events

This approximates event-camera output with:
  - per-pixel state machine
  - separate ON/OFF thresholds
  - per-pixel refractory period
  - event-count aware multi-event emission
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

NS_PER_MS = 1_000_000

STATE_IDLE = 0
STATE_ON = 1
STATE_OFF = -1


def emit_events_state_machine(
    current_log: np.ndarray,
    ref_log: np.ndarray,
    next_allowed_ns: np.ndarray,
    pixel_state: np.ndarray,
    now_ns: int,
    on_threshold: float,
    off_threshold: float,
    refractory_ns: int,
):
    """
    Per-pixel event state machine.

    A pixel can emit:
      - ON events when current_log - ref_log >= on_threshold
      - OFF events when current_log - ref_log <= -off_threshold

    Multi-event emission:
      n_on  = floor((current_log - ref_log) / on_threshold)
      n_off = floor((ref_log - current_log) / off_threshold)

    Refractory:
      A pixel may emit only if now_ns >= next_allowed_ns[pixel].

    State:
      pixel_state stores the last emitted polarity:
        0 = idle
        +1 = last event was ON
        -1 = last event was OFF
    """
    eligible = now_ns >= next_allowed_ns
    if not np.any(eligible):
        zeros = np.zeros_like(ref_log, dtype=np.int32)
        return zeros, zeros, ref_log, next_allowed_ns, pixel_state

    delta = current_log - ref_log

    pos_desired = np.floor(np.maximum(delta, 0.0) / on_threshold).astype(np.int32)
    neg_desired = np.floor(np.maximum(-delta, 0.0) / off_threshold).astype(np.int32)

    pos_emit = np.where(eligible, pos_desired, 0)
    neg_emit = np.where(eligible, neg_desired, 0)

    # Update the internal reference by the number of events actually emitted.
    if np.any(pos_emit):
        ref_log += pos_emit.astype(np.float32) * on_threshold
    if np.any(neg_emit):
        ref_log -= neg_emit.astype(np.float32) * off_threshold

    fired = (pos_emit > 0) | (neg_emit > 0)
    if np.any(fired):
        if refractory_ns > 0:
            next_allowed_ns[fired] = now_ns + refractory_ns

        # Update the explicit state machine polarity.
        pixel_state[fired] = np.where(pos_emit[fired] > 0, STATE_ON, STATE_OFF).astype(np.int8)

    return pos_emit, neg_emit, ref_log, next_allowed_ns, pixel_state


def render_event_panel(acc_pos: np.ndarray, acc_neg: np.ndarray) -> np.ndarray:
    """
    Render accumulated pseudo-events on a white background.

    Positive counts -> green
    Negative counts -> red
    """
    h, w = acc_pos.shape
    frame = np.full((h, w, 3), 255, dtype=np.uint8)

    # Simple saturation map
    pos_alpha = np.clip(acc_pos * 40, 0, 255).astype(np.uint8)
    neg_alpha = np.clip(acc_neg * 40, 0, 255).astype(np.uint8)

    pos_mask = acc_pos > 0
    neg_mask = acc_neg > 0

    # Positive -> green
    frame[pos_mask, 0] = 255 - pos_alpha[pos_mask]  # B
    frame[pos_mask, 2] = 255 - pos_alpha[pos_mask]  # R

    # Negative -> red
    frame[neg_mask, 0] = 255 - neg_alpha[neg_mask]  # B
    frame[neg_mask, 1] = 255 - neg_alpha[neg_mask]  # G

    return frame


def add_panel_labels(image: np.ndarray, split_x: int) -> np.ndarray:
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(out, "Camera", (10, 30), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "Pseudo-events", (split_x + 10, 30), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Real-time pseudo-event reconstruction from webcam video."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="optional resize width for processing/display",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="optional resize height for processing/display",
    )
    parser.add_argument(
        "--on-threshold",
        type=float,
        default=0.2,
        help="ON event log-intensity threshold (default: 0.2)",
    )
    parser.add_argument(
        "--off-threshold",
        type=float,
        default=0.2,
        help="OFF event log-intensity threshold (default: 0.2)",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=50.0,
        help="temporal accumulation window for displayed events (default: 50 ms)",
    )
    parser.add_argument(
        "--refractory-ms",
        type=float,
        default=5.0,
        help="per-pixel refractory period in milliseconds (default: 5 ms)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="epsilon added before log (default: 1e-3)",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="flip camera horizontally for a selfie view",
    )
    args = parser.parse_args(argv)

    if args.on_threshold <= 0:
        raise ValueError("--on-threshold must be positive")
    if args.off_threshold <= 0:
        raise ValueError("--off-threshold must be positive")
    if args.window_ms <= 0:
        raise ValueError("--window-ms must be positive")
    if args.refractory_ms < 0:
        raise ValueError("--refractory-ms must be non-negative")
    if args.eps <= 0:
        raise ValueError("--eps must be positive")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise IOError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Could not read initial frame from camera")

    if args.flip:
        frame = cv2.flip(frame, 1)

    if args.resize_width is not None or args.resize_height is not None:
        target_w = args.resize_width if args.resize_width is not None else frame.shape[1]
        target_h = args.resize_height if args.resize_height is not None else frame.shape[0]
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    ref_log = np.log(gray_f + args.eps)

    # Explicit state machine buffers.
    next_allowed_ns = np.full((h, w), -10**18, dtype=np.int64)
    pixel_state = np.zeros((h, w), dtype=np.int8)

    refractory_ns = int(args.refractory_ms * NS_PER_MS)
    window_ns = int(args.window_ms * NS_PER_MS)

    # Rolling event packets for display window.
    packets = deque()

    # Accumulated event counts inside the current display window.
    acc_pos = np.zeros((h, w), dtype=np.int32)
    acc_neg = np.zeros((h, w), dtype=np.int32)

    cv2.namedWindow("Webcam | Pseudo-events", cv2.WINDOW_NORMAL)

    last_fps_t = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        if args.resize_width is not None or args.resize_height is not None:
            target_w = args.resize_width if args.resize_width is not None else frame.shape[1]
            target_h = args.resize_height if args.resize_height is not None else frame.shape[0]
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0
        current_log = np.log(gray_f + args.eps)

        now_ns = time.perf_counter_ns()

        pos_emit, neg_emit, ref_log, next_allowed_ns, pixel_state = emit_events_state_machine(
            current_log=current_log,
            ref_log=ref_log,
            next_allowed_ns=next_allowed_ns,
            pixel_state=pixel_state,
            now_ns=now_ns,
            on_threshold=args.on_threshold,
            off_threshold=args.off_threshold,
            refractory_ns=refractory_ns,
        )

        # Update rolling accumulation.
        acc_pos += pos_emit
        acc_neg += neg_emit
        packets.append((now_ns, pos_emit, neg_emit))

        while packets and (now_ns - packets[0][0]) > window_ns:
            _, old_pos, old_neg = packets.popleft()
            acc_pos -= old_pos
            acc_neg -= old_neg

        event_panel = render_event_panel(acc_pos, acc_neg)

        # Side-by-side display: camera left, events right.
        combined = np.concatenate([frame, event_panel], axis=1)
        combined = add_panel_labels(combined, split_x=w)

        # FPS display
        t = time.perf_counter()
        dt = t - last_fps_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_fps_t = t

        cv2.putText(
            combined,
            f"FPS: {fps:.1f}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Webcam | Simulated-events", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())