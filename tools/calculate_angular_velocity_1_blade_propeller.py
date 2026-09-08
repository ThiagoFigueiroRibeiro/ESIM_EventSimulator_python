#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

def find_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None


def load_events(npz_path):
    npz = np.load(npz_path)

    time_key = find_key(npz, ["time", "t", "timestamp", "timestamps"])
    x_key = find_key(npz, ["x"])
    y_key = find_key(npz, ["y"])

    if time_key is None or x_key is None or y_key is None:
        keys = list(npz.keys())
        raise KeyError(
            f"Could not find required keys in {npz_path}.\n"
            f"Found keys: {keys}\n"
            f"Expected something like time/t/timestamp and x,y."
        )

    time = np.asarray(npz[time_key])
    x = np.asarray(npz[x_key])
    y = np.asarray(npz[y_key])

    if time.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected time, x, y to be 1D arrays.")
    if not (len(time) == len(x) == len(y)):
        raise ValueError("Expected time, x, y arrays to have the same length.")
    if len(time) == 0:
        raise ValueError("No events found in file.")

    if len(time) >= 2 and np.any(time[1:] < time[:-1]):
        raise ValueError("Timestamps appear not to be sorted. This script assumes sorted by time.")

    return time, x, y, (time_key, x_key, y_key)


def angle_from_origin(px, py, ox=320.0, oy=320.0):
    """
    Angle in [0, 2pi) with origin at (ox, oy).

    Since the plots use inverted y-axis, we define:
      0      -> +x direction
      pi/2   -> upward in the image
      pi     -> -x direction
      3pi/2  -> downward in the image
    """
    ang = np.arctan2(oy - py, px - ox)
    return float(ang % (2.0 * np.pi))


def pca_2d(x, y):
    """
    Returns:
      mu (2,), vals (2,), vecs (2,2)

    vecs[:, 0] = PC1 direction
    vecs[:, 1] = PC2 direction
    vals sorted descending
    """
    pts = np.column_stack([x, y])
    mu = pts.mean(axis=0)
    xc = pts - mu

    C = (xc.T @ xc) / max(len(pts), 1)
    vals, vecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(vals)[::-1]   # descending
    vals = vals[order]
    vecs = vecs[:, order]
    return mu, vals, vecs


def pca_center_2d(x, y):
    """
    Compute a PCA-based center:
      1) project points into PCA coordinates
      2) take the median along PC1 and PC2
      3) map back to original coordinates

    This gives a robust center aligned with the PCA axes.
    """
    pts = np.column_stack([x, y])
    mu, vals, vecs = pca_2d(x, y)

    coords = (pts - mu) @ vecs

    center_pc = np.array([
        np.median(coords[:, 0]),
        np.median(coords[:, 1]),
    ])

    center = mu + center_pc @ vecs.T
    return center, mu, vals, vecs


def compute_angular_velocity(bin_timestamps, angles):
    """
    Compute instantaneous angular velocity using consecutive valid angle samples.
    Returns angular velocity in rad / timestamp_unit.
    """
    vel = np.full_like(angles, np.nan, dtype=float)

    valid_idx = np.where(~np.isnan(angles) & ~np.isnan(bin_timestamps))[0]
    if len(valid_idx) < 2:
        return vel

    t_valid = bin_timestamps[valid_idx]
    a_valid_unwrapped = np.unwrap(angles[valid_idx])

    dt = np.diff(t_valid)
    da = np.diff(a_valid_unwrapped)

    valid_dt = dt != 0.0
    out_idx = valid_idx[1:][valid_dt]
    vel[out_idx] = da[valid_dt] / dt[valid_dt]

    return vel


def get_bin_slice(time_s, edges_s, bin_idx, fps):
    left = edges_s[bin_idx]
    right = edges_s[bin_idx + 1]

    idx_start = np.searchsorted(time_s, left, side="left")
    if bin_idx < fps - 1:
        idx_end = np.searchsorted(time_s, right, side="left")
    else:
        idx_end = np.searchsorted(time_s, right, side="right")
    return idx_start, idx_end


def collect_bin_events(time_s, x, y, edges_s, fps, i, first_events):
    """
    Collect first first_events events from bin i only.
    """
    idx_start, idx_end = get_bin_slice(time_s, edges_s, i, fps)
    take_end = min(idx_start + first_events, idx_end)

    if take_end <= idx_start:
        return np.array([]), np.array([]), np.array([])

    xs = x[idx_start:take_end]
    ys = y[idx_start:take_end]
    ts = time_s[idx_start:take_end]
    return xs, ys, ts


def draw_single_arrow(ax, p0, p1, color, lw=3, mutation_scale=14):
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="->",
        color=color,
        linewidth=lw,
        mutation_scale=mutation_scale,
        shrinkA=0,
        shrinkB=0,
        clip_on=False
    )
    ax.add_patch(arrow)
    return arrow


def save_barycenter_frame(
    out_path,
    xs,
    ys,
    xb,
    yb,
    origin_x,
    origin_y,
    angle_rad,
    ang_vel_rad_ms,
    bin_id,
    timestamp_ms,
    xlim,
    ylim,
    dpi=200
):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot events used in this bin
    if xs is not None and len(xs) > 0:
        ax.scatter(xs, ys, s=18, c="gray", alpha=0.8, label="events")

    # Plot origin
    ax.scatter([origin_x], [origin_y], s=80, c="red", marker="x", linewidths=2.5, label="origin")

    # Plot barycenter and arrow from origin
    if xb is not None and yb is not None:
        ax.scatter([xb], [yb], s=70, c="blue", marker="o", label="barycenter")
        ax.annotate(
            "",
            xy=(xb, yb),
            xytext=(origin_x, origin_y),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2)
        )

        deg = np.degrees(angle_rad)

        if np.isfinite(ang_vel_rad_ms):
            vel_text = f"ω = {ang_vel_rad_ms:.6e} rad/ms"
        else:
            vel_text = "ω = n/a"

        ax.text(
            0.02, 0.98,
            f"θ = {angle_rad:.3f} rad\n{deg:.2f}°\n{vel_text}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none")
        )
    else:
        ax.text(
            0.02, 0.98,
            "No valid barycenter in this bin",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none")
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Bin {bin_id} | t={timestamp_ms:.6f} ms")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_pca_frame(
    out_path,
    xs,
    ys,
    pcx,
    pcy,
    v1,
    v2,
    half_len_pc1,
    half_len_pc2,
    origin_x,
    origin_y,
    angle_rad,
    ang_vel_rad_ms,
    frame_id,
    timestamp_ms,
    xlim,
    ylim,
    plot_events=True,
    events_marker_size=8.0,
    arrow_lw=3.0,
    arrow_mutation_scale=18.0,
    dpi=200
):
    fig, ax = plt.subplots(figsize=(6, 6))

    if xs is not None and len(xs) > 0 and plot_events:
        ax.scatter(xs, ys, s=events_marker_size, c="gray", alpha=0.8, label="events")

    # Origin
    ax.scatter([origin_x], [origin_y], s=80, c="red", marker="x", linewidths=2.5, label="origin")

    if pcx is not None and pcy is not None:
        # PCA center
        ax.scatter([pcx], [pcy], s=70, c="blue", marker="o", label="PCA center")

        # Arrow from origin to PCA center
        ax.annotate(
            "",
            xy=(pcx, pcy),
            xytext=(origin_x, origin_y),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2)
        )

        # PCA axes
        if v1 is not None and v2 is not None:
            p1_start = (pcx, pcy)
            p1_end = (pcx + half_len_pc1 * v1[0], pcy + half_len_pc1 * v1[1])
            draw_single_arrow(
                ax, p1_start, p1_end,
                color="red",
                lw=arrow_lw,
                mutation_scale=arrow_mutation_scale
            )

            p2_start = (pcx, pcy)
            p2_end = (pcx + half_len_pc2 * v2[0], pcy + half_len_pc2 * v2[1])
            draw_single_arrow(
                ax, p2_start, p2_end,
                color="green",
                lw=arrow_lw,
                mutation_scale=arrow_mutation_scale
            )

        deg = np.degrees(angle_rad)
        vel_text = f"ω = {ang_vel_rad_ms:.6e} rad/ms" if np.isfinite(ang_vel_rad_ms) else "ω = n/a"

        ax.text(
            0.02, 0.98,
            f"θ = {angle_rad:.3f} rad\n{deg:.2f}°\n{vel_text}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none")
        )
    else:
        ax.text(
            0.02, 0.98,
            "No valid PCA center in this bin",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none")
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Bin {frame_id} | t={timestamp_ms:.6f} ms")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_barycenter_tracking(raw_time, x, y, edges_s, args, xlim, ylim):
    tracking_frames_dir = (
        os.path.join(args.frames_base_dir, f"tracking_frames_{args.tracking_method}")
        if args.frames_base_dir else f"tracking_frames_{args.tracking_method}"
    )
    os.makedirs(tracking_frames_dir, exist_ok=True)

    print(f"Saving tracking frames to: {tracking_frames_dir}")

    # Per-bin storage
    bin_ids = np.arange(args.fps, dtype=int)
    bin_timestamps_ms = np.full(args.fps, np.nan, dtype=float)

    bary_xs = np.full(args.fps, np.nan, dtype=float)
    bary_ys = np.full(args.fps, np.nan, dtype=float)
    bary_angles = np.full(args.fps, np.nan, dtype=float)
    bary_ang_vel_rad_s = np.full(args.fps, np.nan, dtype=float)
    n_events_per_bin = np.zeros(args.fps, dtype=int)

    # Store per-bin event subsets for frame generation
    xs_frames = [None] * args.fps
    ys_frames = [None] * args.fps

    # First pass: compute barycenters and angular positions
    for i in range(args.fps):
        left = edges_s[i]
        right = edges_s[i + 1]

        idx_start = np.searchsorted(raw_time * 1e-6, left, side="left")
        if i < args.fps - 1:
            idx_end = np.searchsorted(raw_time * 1e-6, right, side="left")
        else:
            idx_end = np.searchsorted(raw_time * 1e-6, right, side="right")

        take_end = min(idx_start + args.first_events, idx_end)

        # Use bin center as display/reference time
        bin_timestamps_s = float(0.5 * (left + right))
        bin_timestamps_ms[i] = bin_timestamps_s * 1e3

        if take_end <= idx_start:
            continue

        xs = x[idx_start:take_end]
        ys = y[idx_start:take_end]

        xb = float(np.mean(xs))
        yb = float(np.mean(ys))
        bary_angle = angle_from_origin(xb, yb, args.origin_x, args.origin_y)

        bary_xs[i] = xb
        bary_ys[i] = yb
        bary_angles[i] = bary_angle
        n_events_per_bin[i] = take_end - idx_start

        xs_frames[i] = xs
        ys_frames[i] = ys

    # Compute angular velocity in rad/s, then convert to rad/ms for display/output
    bary_ang_vel_rad_s = compute_angular_velocity(bin_timestamps_ms * 1e-3, bary_angles)
    bary_ang_vel_rad_ms = bary_ang_vel_rad_s * 1e-3

    # Save frames
    for i in tqdm(range(args.fps), desc="Generating tracking frames", unit="frame"):
        frame_path = os.path.join(
            tracking_frames_dir,
            f"tracking_frame_{i:05d}.{args.image_ext}"
        )
        save_barycenter_frame(
            out_path=frame_path,
            xs=xs_frames[i],
            ys=ys_frames[i],
            xb=bary_xs[i] if np.isfinite(bary_xs[i]) else None,
            yb=bary_ys[i] if np.isfinite(bary_ys[i]) else None,
            origin_x=args.origin_x,
            origin_y=args.origin_y,
            angle_rad=bary_angles[i] if np.isfinite(bary_angles[i]) else 0.0,
            ang_vel_rad_ms=bary_ang_vel_rad_ms[i] if np.isfinite(bary_ang_vel_rad_ms[i]) else np.nan,
            bin_id=i,
            timestamp_ms=bin_timestamps_ms[i],
            xlim=xlim,
            ylim=ylim,
            dpi=args.dpi
        )

    valid_angle = ~np.isnan(bary_angles)
    valid_vel = ~np.isnan(bary_ang_vel_rad_ms)

    if not np.any(valid_angle):
        print("No valid bins with events. Nothing to plot.")
        return

    print()
    print("Angular position stats")
    print("----------------------")
    print(f"Valid bins: {int(np.sum(valid_angle))} / {args.fps}")

    print()
    print("Angular velocity stats")
    print("----------------------")
    if np.any(valid_vel):
        omega_valid = bary_ang_vel_rad_ms[valid_vel]
        mean_omega = float(np.mean(omega_valid))
        mean_omega_deg = float(np.degrees(mean_omega))  # deg/ms

        mae_omega = float(np.mean(np.abs(omega_valid - mean_omega)))
        rmse_omega = float(np.sqrt(np.mean((omega_valid - mean_omega) ** 2)))

        print(f"Valid velocity bins: {int(np.sum(valid_vel))} / {args.fps}")
        print(f"Mean omega: {mean_omega:.6e} rad/ms  ({mean_omega_deg:.6e} deg/ms)")
        print(f"MAE omega:  {mae_omega:.6e} rad/ms")
        print(f"RMSE omega: {rmse_omega:.6e} rad/ms")
    else:
        mean_omega = np.nan
        mean_omega_deg = np.nan
        mae_omega = np.nan
        rmse_omega = np.nan
        print("Not enough valid bins to compute angular velocity.")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "angular_position_velocity_per_timestep.csv")
    csv_data = np.column_stack([
        bin_ids,
        bin_timestamps_ms,
        n_events_per_bin,
        bary_xs,
        bary_ys,
        bary_angles,
        bary_ang_vel_rad_ms,
    ])

    csv_header = (
        "bin,timestamp_ms,n_events,"
        "barycenter_x,barycenter_y,"
        "barycenter_angle_rad,barycenter_angular_velocity_rad_ms"
    )

    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=csv_header,
        comments="",
        fmt=[
            "%d", "%.10f", "%d",
            "%.10f", "%.10f",
            "%.10f", "%.10e"
        ]
    )
    print(f"Saved CSV to: {csv_path}")

    # Plot angular position over time
    pos_plot_path = os.path.join(args.output_dir, f"angular_position_over_time.{args.image_ext}")
    fig, ax = plt.subplots(figsize=(10, 5))

    if np.any(valid_angle):
        ax.plot(
            bin_timestamps_ms[valid_angle],
            np.unwrap(bary_angles[valid_angle]),
            marker="o",
            linewidth=1.5,
            markersize=3,
            color="purple",
            label="barycenter angular position"
        )

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("angular position [rad]")
    ax.set_title("Angular position over time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(pos_plot_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved angular position plot to: {pos_plot_path}")

    # Plot angular velocity over time
    vel_plot_path = os.path.join(args.output_dir, f"angular_velocity_over_time.{args.image_ext}")
    fig, ax = plt.subplots(figsize=(10, 5))

    if np.any(valid_vel):
        omega_valid = bary_ang_vel_rad_ms[valid_vel]
        t_valid_ms = bin_timestamps_ms[valid_vel]

        ax.plot(
            t_valid_ms,
            omega_valid,
            marker="o",
            linewidth=1.5,
            markersize=3,
            color="blue",
            label="barycenter angular velocity"
        )

        # Mean line
        ax.axhline(
            mean_omega,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"mean ω = {mean_omega:.3e} rad/ms ({mean_omega_deg:.3e} deg/ms)"
        )

        # Spread around the mean
        ax.fill_between(
            t_valid_ms,
            mean_omega - mae_omega,
            mean_omega + mae_omega,
            color="red",
            alpha=0.12,
            label=f"±MAE = {mae_omega:.3e} rad/ms"
        )
        ax.fill_between(
            t_valid_ms,
            mean_omega - rmse_omega,
            mean_omega + rmse_omega,
            color="orange",
            alpha=0.12,
            label=f"±RMSE = {rmse_omega:.3e} rad/ms"
        )

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("angular velocity [rad/ms]")
    ax.set_title("Angular velocity over time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(vel_plot_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved angular velocity plot to: {vel_plot_path}")


def run_pca_tracking(raw_time, x, y, edges_s, args, xlim, ylim):
    tracking_frames_dir = (
        os.path.join(args.frames_base_dir, f"tracking_frames_{args.tracking_method}")
        if args.frames_base_dir else f"tracking_frames_{args.tracking_method}"
    )
    os.makedirs(tracking_frames_dir, exist_ok=True)

    print(f"Saving tracking frames to: {tracking_frames_dir}")

    # First pass: collect frame data
    frame_data = []
    all_angles = np.full(args.fps, np.nan, dtype=float)
    all_timestamps_ms = np.full(args.fps, np.nan, dtype=float)

    prev_v1 = None
    prev_v2 = None

    for i in range(args.fps):
        xs, ys, ts = collect_bin_events(raw_time * 1e-6, x, y, edges_s, args.fps, i, args.first_events)

        left_s = edges_s[i]
        right_s = edges_s[i + 1]
        bin_t_s = 0.5 * (left_s + right_s)
        bin_t_ms = bin_t_s * 1e3
        all_timestamps_ms[i] = bin_t_ms

        if len(xs) < args.min_points_for_pca:
            frame_data.append({
                "valid": False,
                "xs": xs,
                "ys": ys,
                "pcx": None,
                "pcy": None,
                "v1": None,
                "v2": None,
                "half_len_pc1": None,
                "half_len_pc2": None,
                "angle_rad": np.nan,
                "timestamp_ms": bin_t_ms,
                "bin_id": i,
            })
            continue

        # PCA center and axes: unchanged implementation
        center, mu, vals, vecs = pca_center_2d(xs, ys)
        pcx, pcy = float(center[0]), float(center[1])

        v1 = vecs[:, 0].copy()
        v2 = vecs[:, 1].copy()

        # Stabilize sign across frames so arrows do not randomly flip direction
        if prev_v1 is not None and np.dot(v1, prev_v1) < 0:
            v1 = -v1
        if prev_v2 is not None and np.dot(v2, prev_v2) < 0:
            v2 = -v2

        prev_v1 = v1.copy()
        prev_v2 = v2.copy()

        if args.pca_len_mode == "range":
            base = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
            half_len_pc1 = args.pca_len_scale * base * 0.5 * args.pc1_boost
            half_len_pc2 = args.pca_len_scale * base * 0.5 * args.pc2_boost
        else:
            half_len_pc1 = args.pca_len_scale * float(np.sqrt(vals[0] + 1e-12)) * args.pc1_boost
            half_len_pc2 = args.pca_len_scale * float(np.sqrt(vals[1] + 1e-12)) * args.pc2_boost

        angle_rad = angle_from_origin(pcx, pcy, args.origin_x, args.origin_y)
        all_angles[i] = angle_rad

        frame_data.append({
            "valid": True,
            "xs": xs,
            "ys": ys,
            "pcx": pcx,
            "pcy": pcy,
            "v1": v1,
            "v2": v2,
            "half_len_pc1": half_len_pc1,
            "half_len_pc2": half_len_pc2,
            "angle_rad": angle_rad,
            "timestamp_ms": bin_t_ms,
            "bin_id": i,
        })

    # Compute angular velocity after all angles are known
    full_vel = compute_angular_velocity(all_timestamps_ms, all_angles)

    valid_angle = ~np.isnan(all_angles)
    valid_vel = ~np.isnan(full_vel)

    if not np.any(valid_angle):
        print("No valid PCA frames. Nothing to plot.")
        return

    # Second pass: save frames with computed velocity
    for fd in tqdm(frame_data, desc="Generating tracking frames", unit="frame"):
        i = fd["bin_id"]
        out_path = os.path.join(
            tracking_frames_dir,
            f"{args.frame_prefix}_{i:04d}.{args.image_ext}"
        )

        ang_vel = full_vel[i] if fd["valid"] else np.nan

        save_pca_frame(
            out_path=out_path,
            xs=fd["xs"],
            ys=fd["ys"],
            pcx=fd["pcx"],
            pcy=fd["pcy"],
            v1=fd["v1"],
            v2=fd["v2"],
            half_len_pc1=fd["half_len_pc1"],
            half_len_pc2=fd["half_len_pc2"],
            origin_x=args.origin_x,
            origin_y=args.origin_y,
            angle_rad=fd["angle_rad"] if fd["valid"] else 0.0,
            ang_vel_rad_ms=ang_vel,
            frame_id=i,
            timestamp_ms=fd["timestamp_ms"],
            xlim=xlim,
            ylim=ylim,
            plot_events=args.plot_events,
            events_marker_size=args.events_marker_size,
            arrow_lw=args.arrow_lw,
            arrow_mutation_scale=args.arrow_mutation_scale,
            dpi=args.dpi
        )

    # Stats
    print()
    print("Angular position stats")
    print("----------------------")
    print(f"Valid bins: {int(np.sum(valid_angle))} / {args.fps}")

    print()
    print("Angular velocity stats")
    print("----------------------")
    if np.any(valid_vel):
        omega_valid = full_vel[valid_vel]
        mean_omega = float(np.mean(omega_valid))
        mean_omega_deg = float(np.degrees(mean_omega))
        mae_omega = float(np.mean(np.abs(omega_valid - mean_omega)))
        rmse_omega = float(np.sqrt(np.mean((omega_valid - mean_omega) ** 2)))

        print(f"Valid velocity bins: {int(np.sum(valid_vel))} / {args.fps}")
        print(f"Mean omega: {mean_omega:.6e} rad/ms  ({mean_omega_deg:.6e} deg/ms)")
        print(f"MAE omega:  {mae_omega:.6e} rad/ms")
        print(f"RMSE omega: {rmse_omega:.6e} rad/ms")
    else:
        mean_omega = np.nan
        mean_omega_deg = np.nan
        mae_omega = np.nan
        rmse_omega = np.nan
        print("Not enough valid bins to compute angular velocity.")

    # Build full-size arrays for CSV
    pca_x_full = np.full(args.fps, np.nan, dtype=float)
    pca_y_full = np.full(args.fps, np.nan, dtype=float)
    for fd in frame_data:
        if fd["valid"]:
            pca_x_full[fd["bin_id"]] = fd["pcx"]
            pca_y_full[fd["bin_id"]] = fd["pcy"]

    # Save CSV
    csv_path = os.path.join(args.output_dir, "angular_position_velocity_per_timestep.csv")
    csv_data = np.column_stack([
        np.arange(args.fps, dtype=int),
        all_timestamps_ms,
        valid_angle.astype(int),
        pca_x_full,
        pca_y_full,
        all_angles,
        full_vel,
    ])

    csv_header = (
        "frame,timestamp_ms,valid,"
        "pca_center_x,pca_center_y,"
        "pca_center_angle_rad,pca_center_angular_velocity_rad_ms"
    )

    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=csv_header,
        comments="",
        fmt=[
            "%d", "%.10f", "%d",
            "%.10f", "%.10f",
            "%.10f", "%.10e"
        ]
    )
    print(f"Saved CSV to: {csv_path}")

    # Plot angular position over time
    pos_plot_path = os.path.join(args.output_dir, f"angular_position_over_time.{args.image_ext}")
    fig, ax = plt.subplots(figsize=(10, 5))
    if np.any(valid_angle):
        ax.plot(
            all_timestamps_ms[valid_angle],
            np.unwrap(all_angles[valid_angle]),
            marker="o",
            linewidth=1.5,
            markersize=3,
            color="purple",
            label="PCA center angular position"
        )
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("angular position [rad]")
    ax.set_title("Angular position over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pos_plot_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved angular position plot to: {pos_plot_path}")

    # Plot angular velocity over time
    vel_plot_path = os.path.join(args.output_dir, f"angular_velocity_over_time.{args.image_ext}")
    fig, ax = plt.subplots(figsize=(10, 5))
    if np.any(valid_vel):
        omega_valid = full_vel[valid_vel]
        t_valid_ms = all_timestamps_ms[valid_vel]

        ax.plot(
            t_valid_ms,
            omega_valid,
            marker="o",
            linewidth=1.5,
            markersize=3,
            color="blue",
            label="PCA center angular velocity"
        )

        ax.axhline(
            mean_omega,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"mean ω = {mean_omega:.3e} rad/ms ({mean_omega_deg:.3e} deg/ms)"
        )

        ax.fill_between(
            t_valid_ms,
            mean_omega - mae_omega,
            mean_omega + mae_omega,
            color="red",
            alpha=0.12,
            label=f"±MAE = {mae_omega:.3e} rad/ms"
        )

        ax.fill_between(
            t_valid_ms,
            mean_omega - rmse_omega,
            mean_omega + rmse_omega,
            color="orange",
            alpha=0.12,
            label=f"±RMSE = {rmse_omega:.3e} rad/ms"
        )

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("angular velocity [rad/ms]")
    ax.set_title("Angular velocity over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(vel_plot_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved angular velocity plot to: {vel_plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="events.npz", help="Path to events.npz")
    parser.add_argument("--fps", type=int, required=True, help="Number of equally spaced time bins/frames.")
    parser.add_argument("--first_events", type=int, default=50, help="How many first events to use per time bin.")

    parser.add_argument(
        "--tracking_method",
        type=str,
        default="barycenter",
        choices=["barycenter", "pca"],
        help="Event tracking method."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tracking_output",
        help="Output directory for CSV and plots."
    )

    parser.add_argument(
        "--frames_base_dir",
        type=str,
        default="",
        help="Base directory for tracking_frames_[method]. Empty means current directory."
    )

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--image_ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--frame_prefix", type=str, default="tracking_frame")

    parser.add_argument("--plot_events", action="store_true", default=True, help="Plot events in the frame.")
    parser.add_argument("--no_plot_events", dest="plot_events", action="store_false", help="Disable event plotting.")
    parser.add_argument("--events_marker_size", type=float, default=8.0)

    parser.add_argument("--min_points_for_pca", type=int, default=3,
                        help="If fewer than this number of points exist, PCA is skipped.")

    parser.add_argument("--pca_len_mode", type=str, default="sqrt_eig",
                        choices=["range", "sqrt_eig"],
                        help="Scaling mode for PCA arrows.")
    parser.add_argument("--pca_len_scale", type=float, default=0.8,
                        help="Global multiplicative factor for PCA arrow lengths.")
    parser.add_argument("--pc1_boost", type=float, default=2.5,
                        help="Extra multiplier for PC1 red arrow length.")
    parser.add_argument("--pc2_boost", type=float, default=1.0,
                        help="Extra multiplier for PC2 green arrow length.")
    parser.add_argument("--arrow_mutation_scale", type=float, default=18.0,
                        help="Arrowhead size.")
    parser.add_argument("--arrow_lw", type=float, default=3.0,
                        help="Arrow line width.")

    parser.add_argument("--show", action="store_true", help="Show figures while generating.")

    # Reference origin for angular computation
    parser.add_argument("--origin_x", type=float, default=320.0,
                        help="Origin x for angular position calculation.")
    parser.add_argument("--origin_y", type=float, default=320.0,
                        help="Origin y for angular position calculation.")

    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.first_events <= 0:
        raise ValueError("--first_events must be > 0")

    raw_time, x, y, keys = load_events(args.npz)

    # raw timestamps are in microseconds
    time_s = raw_time * 1e-6
    time_ms = raw_time * 1e-3

    # Method-specific output folder
    args.output_dir = f"{args.output_dir}_{args.tracking_method.upper()}"
    os.makedirs(args.output_dir, exist_ok=True)

    first_ts_s = float(time_s[0])
    last_ts_s = float(time_s[-1])
    if last_ts_s <= first_ts_s:
        raise ValueError("Invalid timestamp range: last_ts must be greater than first_ts.")

    edges_s = np.linspace(first_ts_s, last_ts_s, args.fps + 1)

    x_min = float(min(np.min(x), args.origin_x))
    x_max = float(max(np.max(x), args.origin_x))
    y_min = float(min(np.min(y), args.origin_y))
    y_max = float(max(np.max(y), args.origin_y))

    pad_x = 10.0
    pad_y = 10.0
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    print(f"Loaded keys: time={keys[0]}, x={keys[1]}, y={keys[2]}")
    print(f"Tracking method: {args.tracking_method}")
    print(f"Saving outputs to: {args.output_dir}")
    print(f"Angular origin: ({args.origin_x}, {args.origin_y})")
    print("Timestamp scale: microseconds -> seconds for computation (x1e-6)")
    print("Timestamp display: milliseconds (x1e-3)")
    print("Angular velocity display: rad/ms and deg/ms")
    print(f"Using {args.fps} time bins and first {args.first_events} events per bin")

    if args.tracking_method == "barycenter":
        print("Selected tracking mode: barycenter")
        run_barycenter_tracking(raw_time, x, y, edges_s, args, xlim, ylim)
    elif args.tracking_method == "pca":
        print("Selected tracking mode: pca")
        run_pca_tracking(raw_time, x, y, edges_s, args, xlim, ylim)
    else:
        raise ValueError(f"Unknown tracking method: {args.tracking_method}")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)