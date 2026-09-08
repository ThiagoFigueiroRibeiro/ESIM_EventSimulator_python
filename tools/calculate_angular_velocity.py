
#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm


# -----------------------------
# IO
# -----------------------------
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


# -----------------------------
# Geometry / PCA
# -----------------------------
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


def estimate_global_center(x, y, method="pca_center"):
    """
    Estimate the global center once and reuse it.

    method:
      - pca_center: PCA-based robust center
      - mean: global mean
      - median: global median
    """
    method = method.lower()

    if method == "mean":
        return float(np.mean(x)), float(np.mean(y))
    elif method == "median":
        return float(np.median(x)), float(np.median(y))
    elif method == "pca_center":
        center, _, _, _ = pca_center_2d(x, y)
        return float(center[0]), float(center[1])
    else:
        raise ValueError(f"Unknown center method: {method}")


def principal_axis_angle_from_vec(v):
    """
    Convert a 2D vector to a signed angle in image coordinates.
    Here we use atan2(-vy, vx) to match the image convention.
    """
    return float(np.arctan2(-v[1], v[0]))


def wrap_to_pi(theta):
    """Wrap angle to (-pi, pi]."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def unwrap_continuous_angle(theta_obs, prev_theta=None):
    """
    Standard 2*pi unwrapping for signed angles.
    """
    if prev_theta is None or not np.isfinite(prev_theta):
        return float(theta_obs)
    return float(prev_theta + wrap_to_pi(theta_obs - prev_theta))


def choose_axis_sign(v1, prev_v1_signed=None, origin_x=None, origin_y=None,
                     local_cx=None, local_cy=None):
    """
    Resolve PCA sign ambiguity.

    Priority:
      1) If a previous signed axis exists, keep continuity with it.
      2) Otherwise use the vector from global origin to local centroid.
      3) Otherwise keep the raw PCA sign.
    """
    # 1) Temporal continuity wins
    if prev_v1_signed is not None and np.all(np.isfinite(prev_v1_signed)):
        return v1 if np.dot(v1, prev_v1_signed) >= 0 else -v1

    # 2) First valid frame: use a spatial reference if available
    if (
        origin_x is not None and origin_y is not None
        and local_cx is not None and local_cy is not None
    ):
        centroid_vec = np.array([local_cx - origin_x, local_cy - origin_y], dtype=float)
        n = np.linalg.norm(centroid_vec)
        if n > 1e-12:
            centroid_vec /= n
            return v1 if np.dot(v1, centroid_vec) >= 0 else -v1

    # 3) Fallback
    return v1


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


# -----------------------------
# Plotting
# -----------------------------
def save_pca_frame(
    out_path,
    xs,
    ys,
    local_cx,
    local_cy,
    global_origin_x,
    global_origin_y,
    v1,
    v2,
    half_len_pc1,
    half_len_pc2,
    theta_obs,
    theta_unwrapped,
    omega_rad_ms,
    bin_id,
    t_start_ms,
    t_end_ms,
    t_center_ms,
    n_events,
    xlim,
    ylim,
    plot_events=True,
    events_marker_size=8.0,
    arrow_lw=3.0,
    arrow_mutation_scale=18.0,
    dpi=200,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    if xs is not None and len(xs) > 0 and plot_events:
        ax.scatter(xs, ys, s=events_marker_size, c="gray", alpha=0.8, label="events")

    # Global propeller/metronome pivot center
    ax.scatter(
        [global_origin_x], [global_origin_y],
        s=80, c="red", marker="x", linewidths=2.5, label="global center"
    )

    # Local centroid / anchor for PCA visualization
    if local_cx is not None and local_cy is not None:
        ax.scatter([local_cx], [local_cy], s=70, c="blue", marker="o", label="bin centroid")

        ax.annotate(
            "",
            xy=(local_cx, local_cy),
            xytext=(global_origin_x, global_origin_y),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2)
        )

        if v1 is not None and v2 is not None and half_len_pc1 is not None and half_len_pc2 is not None:
            p1_start = (local_cx, local_cy)
            p1_end = (local_cx + half_len_pc1 * v1[0], local_cy + half_len_pc1 * v1[1])
            draw_single_arrow(
                ax, p1_start, p1_end,
                color="red",
                lw=arrow_lw,
                mutation_scale=arrow_mutation_scale
            )

            p2_start = (local_cx, local_cy)
            p2_end = (local_cx + half_len_pc2 * v2[0], local_cy + half_len_pc2 * v2[1])
            draw_single_arrow(
                ax, p2_start, p2_end,
                color="green",
                lw=arrow_lw,
                mutation_scale=arrow_mutation_scale
            )

        if np.isfinite(theta_unwrapped):
            theta_unwrapped_deg = np.degrees(theta_unwrapped)
            vel_text = f"ω = {omega_rad_ms:.6e} rad/ms" if np.isfinite(omega_rad_ms) else "ω = n/a"

            ax.text(
                0.02, 0.98,
                f"θ = {theta_unwrapped:.3f} rad ({theta_unwrapped_deg:.2f}°)\n"
                f"{vel_text}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9.5,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="none")
            )
        else:
            ax.text(
                0.02, 0.98,
                f"bin = {bin_id}\n"
                f"events = {n_events}\n"
                f"t = {t_center_ms:.6f} ms\n"
                f"No valid PCA measurement",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="none")
            )
    else:
        ax.text(
            0.02, 0.98,
            f"bin = {bin_id}\n"
            f"events = {n_events}\n"
            f"t = {t_center_ms:.6f} ms\n"
            f"No valid events",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="none")
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Bin {bin_id} | {t_start_ms:.6f} ms → {t_end_ms:.6f} ms")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# -----------------------------
# Tracking pipeline
# -----------------------------
def run_pca_tracking(raw_time, x, y, args, xlim, ylim):
    """
    Split all events into equal-count bins.
    For each bin, use only the first `analysis_events_per_bin` events.

    For each bin:
      - PCA on those selected events
      - resolve PCA sign using previous signed direction (primary)
      - extract a signed angle in [-pi, pi)
      - unwrap continuously through time
      - estimate angular velocity from consecutive unwrapped angles
    """
    global_cx, global_cy = estimate_global_center(x, y, method=args.center_method)

    if args.origin_x is None:
        args.origin_x = global_cx
    if args.origin_y is None:
        args.origin_y = global_cy

    tracking_frames_dir = (
        os.path.join(args.frames_base_dir, "tracking_frames_pca")
        if args.frames_base_dir else "tracking_frames_pca"
    )
    os.makedirs(tracking_frames_dir, exist_ok=True)

    print(f"Saving tracking frames to: {tracking_frames_dir}")
    print(f"Global center used as origin: ({args.origin_x:.3f}, {args.origin_y:.3f})")
    print(f"Center estimation method: {args.center_method}")
    print(f"Number of bins: {args.num_bins}")
    print(f"Events used per bin (max): {args.analysis_events_per_bin}")
    print(f"Minimum points for PCA: {args.min_points_for_pca}")

    n_events_total = len(x)
    if args.num_bins <= 0:
        raise ValueError("--num_bins must be > 0")

    # Equal-count bin edges across the full event stream
    bin_edges = np.linspace(0, n_events_total, args.num_bins + 1, dtype=int)
    n_bins = args.num_bins

    # Per-bin storage
    bin_ids = np.arange(n_bins, dtype=int)
    start_event_idx = np.full(n_bins, -1, dtype=int)
    end_event_idx = np.full(n_bins, -1, dtype=int)
    n_events_per_bin = np.zeros(n_bins, dtype=int)

    window_t_ms = np.full(n_bins, np.nan, dtype=float)
    window_t_start_ms = np.full(n_bins, np.nan, dtype=float)
    window_t_end_ms = np.full(n_bins, np.nan, dtype=float)

    local_cx = np.full(n_bins, np.nan, dtype=float)
    local_cy = np.full(n_bins, np.nan, dtype=float)

    pca_x_full = np.full(n_bins, np.nan, dtype=float)
    pca_y_full = np.full(n_bins, np.nan, dtype=float)

    theta_obs = np.full(n_bins, np.nan, dtype=float)
    theta_unwrapped = np.full(n_bins, np.nan, dtype=float)

    omega_rad_s = np.full(n_bins, np.nan, dtype=float)
    omega_rad_ms = np.full(n_bins, np.nan, dtype=float)

    eigval1 = np.full(n_bins, np.nan, dtype=float)
    eigval2 = np.full(n_bins, np.nan, dtype=float)

    valid_measurement = np.zeros(n_bins, dtype=int)
    propagated_only = np.zeros(n_bins, dtype=int)

    prev_v1_signed = None
    prev_theta = None
    prev_omega = None
    prev_time_s = None

    frame_data = []

    for i in range(n_bins):
        bin_s = int(bin_edges[i])
        bin_e = int(bin_edges[i + 1])

        # Use only the first N events of this bin
        e_use = min(bin_s + args.analysis_events_per_bin, bin_e)

        start_event_idx[i] = bin_s
        end_event_idx[i] = e_use

        xs = x[bin_s:e_use]
        ys = y[bin_s:e_use]
        ts = raw_time[bin_s:e_use] * 1e-6  # microseconds -> seconds

        n_events = len(xs)
        n_events_per_bin[i] = n_events

        if n_events > 0:
            window_t_s = float(np.mean(ts))
            window_t_ms[i] = window_t_s * 1e3
            window_t_start_ms[i] = float(ts[0] * 1e3)
            window_t_end_ms[i] = float(ts[-1] * 1e3)

            lc_x = float(np.mean(xs))
            lc_y = float(np.mean(ys))
            local_cx[i] = lc_x
            local_cy[i] = lc_y
        else:
            window_t_s = np.nan
            lc_x = None
            lc_y = None

        if n_events >= args.min_points_for_pca:
            center, mu, vals, vecs = pca_center_2d(xs, ys)

            v1 = vecs[:, 0].copy()
            v2 = vecs[:, 1].copy()

            # Enforce sign continuity with the previous bin
            v1_signed = choose_axis_sign(
                v1,
                prev_v1_signed=prev_v1_signed,
                origin_x=args.origin_x,
                origin_y=args.origin_y,
                local_cx=lc_x,
                local_cy=lc_y
            )

            sign = 1.0 if np.dot(v1_signed, v1) >= 0 else -1.0
            v1_signed = sign * v1
            v2_signed = sign * v2  # keep PCA basis consistent for plotting

            prev_v1_signed = v1_signed.copy()

            if args.pca_len_mode == "range":
                base = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
                half_len_pc1 = args.pca_len_scale * base * 0.5 * args.pc1_boost
                half_len_pc2 = args.pca_len_scale * base * 0.5 * args.pc2_boost
            else:
                half_len_pc1 = args.pca_len_scale * float(np.sqrt(vals[0] + 1e-12)) * args.pc1_boost
                half_len_pc2 = args.pca_len_scale * float(np.sqrt(vals[1] + 1e-12)) * args.pc2_boost

            theta_o = principal_axis_angle_from_vec(v1_signed)

            if prev_theta is None or not np.isfinite(prev_theta):
                theta_u = float(theta_o)
                omega_s = np.nan
            else:
                dt = window_t_s - prev_time_s if prev_time_s is not None and np.isfinite(prev_time_s) else np.nan
                theta_u = unwrap_continuous_angle(theta_o, prev_theta)

                if np.isfinite(dt) and dt > 0:
                    omega_s = (theta_u - prev_theta) / dt
                else:
                    omega_s = np.nan

            valid_measurement[i] = 1
            theta_obs[i] = theta_o
            theta_unwrapped[i] = theta_u
            omega_rad_s[i] = omega_s
            omega_rad_ms[i] = omega_s * 1e-3 if np.isfinite(omega_s) else np.nan
            eigval1[i] = float(vals[0])
            eigval2[i] = float(vals[1])

            pca_x_full[i] = float(center[0])
            pca_y_full[i] = float(center[1])

            prev_theta = theta_u
            prev_omega = omega_s if np.isfinite(omega_s) else prev_omega
            prev_time_s = window_t_s

            frame_data.append({
                "bin_id": i,
                "xs": xs,
                "ys": ys,
                "local_cx": lc_x,
                "local_cy": lc_y,
                "v1": v1_signed,
                "v2": v2_signed,
                "half_len_pc1": half_len_pc1,
                "half_len_pc2": half_len_pc2,
                "theta_obs": theta_o,
                "theta_unwrapped": theta_u,
                "omega_ms": omega_rad_ms[i],
                "t_start_ms": window_t_start_ms[i],
                "t_end_ms": window_t_end_ms[i],
                "t_center_ms": window_t_ms[i],
                "n_events": n_events,
                "valid": True,
            })
        else:
            # Too few points for PCA
            if (
                args.propagate_missing
                and prev_theta is not None
                and np.isfinite(prev_theta)
                and prev_time_s is not None
                and np.isfinite(prev_time_s)
                and n_events > 0
            ):
                dt = window_t_s - prev_time_s
                if np.isfinite(dt) and dt > 0:
                    theta_u = prev_theta + (prev_omega if np.isfinite(prev_omega) else 0.0) * dt
                    omega_s = prev_omega if np.isfinite(prev_omega) else np.nan
                    propagated_only[i] = 1

                    theta_unwrapped[i] = theta_u
                    omega_rad_s[i] = omega_s
                    omega_rad_ms[i] = omega_s * 1e-3 if np.isfinite(omega_s) else np.nan

                    prev_theta = theta_u
                    prev_time_s = window_t_s
                else:
                    theta_u = np.nan
                    omega_s = np.nan
            else:
                theta_u = np.nan
                omega_s = np.nan

            frame_data.append({
                "bin_id": i,
                "xs": xs,
                "ys": ys,
                "local_cx": lc_x,
                "local_cy": lc_y,
                "v1": None,
                "v2": None,
                "half_len_pc1": None,
                "half_len_pc2": None,
                "theta_obs": np.nan,
                "theta_unwrapped": theta_u,
                "omega_ms": omega_rad_ms[i],
                "t_start_ms": window_t_start_ms[i],
                "t_end_ms": window_t_end_ms[i],
                "t_center_ms": window_t_ms[i],
                "n_events": n_events,
                "valid": False,
            })

    # Save frames
    for fd in tqdm(frame_data, desc="Generating tracking frames", unit="frame"):
        i = fd["bin_id"]
        out_path = os.path.join(
            tracking_frames_dir,
            f"{args.frame_prefix}_{i:05d}.{args.image_ext}"
        )

        save_pca_frame(
            out_path=out_path,
            xs=fd["xs"],
            ys=fd["ys"],
            local_cx=fd["local_cx"],
            local_cy=fd["local_cy"],
            global_origin_x=args.origin_x,
            global_origin_y=args.origin_y,
            v1=fd["v1"],
            v2=fd["v2"],
            half_len_pc1=fd["half_len_pc1"],
            half_len_pc2=fd["half_len_pc2"],
            theta_obs=fd["theta_obs"] if np.isfinite(fd["theta_obs"]) else np.nan,
            theta_unwrapped=fd["theta_unwrapped"] if np.isfinite(fd["theta_unwrapped"]) else np.nan,
            omega_rad_ms=fd["omega_ms"] if np.isfinite(fd["omega_ms"]) else np.nan,
            bin_id=i,
            t_start_ms=fd["t_start_ms"],
            t_end_ms=fd["t_end_ms"],
            t_center_ms=fd["t_center_ms"],
            n_events=fd["n_events"],
            xlim=xlim,
            ylim=ylim,
            plot_events=args.plot_events,
            events_marker_size=args.events_marker_size,
            arrow_lw=args.arrow_lw,
            arrow_mutation_scale=args.arrow_mutation_scale,
            dpi=args.dpi
        )

    valid_angle = np.isfinite(theta_unwrapped)
    valid_vel = np.isfinite(omega_rad_ms)

    if not np.any(valid_angle):
        print("No valid tracking bins. Nothing to plot.")
        return

    # -----------------------------
    # Stats
    # -----------------------------
    print()
    print("Angular position stats")
    print("----------------------")
    print(f"Valid bins with angle:     {int(np.sum(valid_angle))} / {n_bins}")
    print(f"Valid PCA measurements:    {int(np.sum(valid_measurement))} / {n_bins}")
    print(f"Propagated-only bins:      {int(np.sum(propagated_only))} / {n_bins}")

    print()
    print("Angular velocity stats")
    print("----------------------")
    if np.any(valid_vel):
        omega_valid = omega_rad_ms[valid_vel]
        mean_omega = float(np.mean(omega_valid))
        mean_omega_deg = float(np.degrees(mean_omega))  # deg/ms

        mae_omega = float(np.mean(np.abs(omega_valid - mean_omega)))
        rmse_omega = float(np.sqrt(np.mean((omega_valid - mean_omega) ** 2)))

        print(f"Valid velocity bins:       {int(np.sum(valid_vel))} / {n_bins}")
        print(f"Mean omega: {mean_omega:.6e} rad/ms  ({mean_omega_deg:.6e} deg/ms)")
        print(f"MAE omega:  {mae_omega:.6e} rad/ms")
        print(f"RMSE omega: {rmse_omega:.6e} rad/ms")
    else:
        mean_omega = np.nan
        mean_omega_deg = np.nan
        mae_omega = np.nan
        rmse_omega = np.nan
        print("Not enough valid bins to compute angular velocity.")

    # -----------------------------
    # CSV
    # -----------------------------
    csv_path = os.path.join(args.output_dir, "angular_position_velocity_per_bin.csv")
    csv_data = np.column_stack([
        bin_ids,
        start_event_idx,
        end_event_idx,
        window_t_start_ms,
        window_t_end_ms,
        window_t_ms,
        n_events_per_bin,
        valid_measurement,
        propagated_only,
        local_cx,
        local_cy,
        pca_x_full,
        pca_y_full,
        theta_obs,
        theta_unwrapped,
        omega_rad_s,
        omega_rad_ms,
        eigval1,
        eigval2,
    ])

    csv_header = (
        "bin,start_event_idx,end_event_idx,"
        "t_start_ms,t_end_ms,t_center_ms,"
        "n_events,valid_measurement,propagated_only,"
        "bin_centroid_x,bin_centroid_y,"
        "pca_center_x,pca_center_y,"
        "theta_obs_rad,theta_unwrapped_rad,"
        "angular_velocity_rad_s,angular_velocity_rad_ms,"
        "eigval1,eigval2"
    )

    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=csv_header,
        comments="",
        fmt=[
            "%d", "%d", "%d",
            "%.10f", "%.10f", "%.10f",
            "%d", "%d", "%d",
            "%.10f", "%.10f",
            "%.10f", "%.10f",
            "%.10f", "%.10f",
            "%.10e", "%.10e",
            "%.10e", "%.10e"
        ]
    )
    print(f"Saved CSV to: {csv_path}")

    # -----------------------------
    # Plot angular position over time
    # -----------------------------
    pos_plot_path = os.path.join(args.output_dir, f"angular_position_over_time.{args.image_ext}")
    fig, ax = plt.subplots(figsize=(10, 5))

    if np.any(valid_angle):
        ax.plot(
            window_t_ms[valid_angle],
            theta_unwrapped[valid_angle],
            marker="o",
            linewidth=1.5,
            markersize=3,
            color="purple",
            label="unwrapped angle"
        )

    meas_idx = valid_measurement.astype(bool) & np.isfinite(theta_unwrapped)
    if np.any(meas_idx):
        ax.scatter(
            window_t_ms[meas_idx],
            theta_unwrapped[meas_idx],
            s=20,
            color="orange",
            alpha=0.8,
            label="PCA-backed bins"
        )

    if np.any(propagated_only.astype(bool)):
        prop_idx = propagated_only.astype(bool) & np.isfinite(theta_unwrapped)
        ax.scatter(
            window_t_ms[prop_idx],
            theta_unwrapped[prop_idx],
            s=20,
            color="gray",
            alpha=0.8,
            label="propagated-only bins"
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

    # -----------------------------
    # Plot angular velocity over time
    # -----------------------------
    vel_plot_path = os.path.join(args.output_dir, f"angular_velocity_over_time.{args.image_ext}")
    fig, ax = plt.subplots(figsize=(10, 5))

    if np.any(valid_vel):
        omega_valid = omega_rad_ms[valid_vel]
        t_valid_ms = window_t_ms[valid_vel]

        ax.plot(
            t_valid_ms,
            omega_valid,
            marker="o",
            linewidth=1.5,
            markersize=3,
            color="blue",
            label="angular velocity"
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PCA-based continuous angle tracker using equal-count bins and the first N events per bin."
    )

    parser.add_argument("--npz", type=str, default="events.npz", help="Path to events.npz")

    parser.add_argument(
        "--num_bins",
        type=int,
        default=240,
        help="Number of equal-count bins across the full event stream."
    )
    parser.add_argument(
        "--analysis_events_per_bin",
        type=int,
        default=400,
        help="Use only the first N events of each bin for PCA/algebra."
    )
    parser.add_argument(
        "--min_points_for_pca",
        type=int,
        default=3,
        help="If fewer than this number of points exist in a bin, PCA is skipped."
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
        help="Base directory for tracking_frames_pca. Empty means current directory."
    )

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--image_ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--frame_prefix", type=str, default="tracking_frame")

    parser.add_argument("--plot_events", action="store_true", default=True, help="Plot events in the frame.")
    parser.add_argument("--no_plot_events", dest="plot_events", action="store_false", help="Disable event plotting.")
    parser.add_argument("--events_marker_size", type=float, default=8.0)

    parser.add_argument(
        "--pca_len_mode",
        type=str,
        default="sqrt_eig",
        choices=["range", "sqrt_eig"],
        help="Scaling mode for PCA arrows."
    )
    parser.add_argument(
        "--pca_len_scale",
        type=float,
        default=0.8,
        help="Global multiplicative factor for PCA arrow lengths."
    )
    parser.add_argument(
        "--pc1_boost",
        type=float,
        default=2.5,
        help="Extra multiplier for PC1 red arrow length."
    )
    parser.add_argument(
        "--pc2_boost",
        type=float,
        default=1.0,
        help="Extra multiplier for PC2 green arrow length."
    )
    parser.add_argument(
        "--arrow_mutation_scale",
        type=float,
        default=18.0,
        help="Arrowhead size."
    )
    parser.add_argument(
        "--arrow_lw",
        type=float,
        default=3.0,
        help="Arrow line width."
    )

    parser.add_argument("--show", action="store_true", help="Show figures while generating.")

    # Global center / origin
    parser.add_argument(
        "--center_method",
        type=str,
        default="pca_center",
        choices=["pca_center", "mean", "median"],
        help="How to estimate the global center once."
    )
    parser.add_argument(
        "--origin_x",
        type=float,
        default=None,
        help="Optional manual global center x. If omitted, the global center is estimated automatically."
    )
    parser.add_argument(
        "--origin_y",
        type=float,
        default=None,
        help="Optional manual global center y. If omitted, the global center is estimated automatically."
    )

    parser.add_argument(
        "--propagate_missing",
        action="store_true",
        default=True,
        help="If a bin lacks enough points for PCA, propagate the state using the previous angular velocity."
    )
    parser.add_argument(
        "--no_propagate_missing",
        dest="propagate_missing",
        action="store_false",
        help="Disable propagation for bins that lack enough points for PCA."
    )

    args = parser.parse_args()

    if args.num_bins <= 0:
        raise ValueError("--num_bins must be > 0")
    if args.analysis_events_per_bin <= 0:
        raise ValueError("--analysis_events_per_bin must be > 0")
    if args.min_points_for_pca <= 0:
        raise ValueError("--min_points_for_pca must be > 0")

    raw_time, x, y, keys = load_events(args.npz)

    # timestamps are assumed microseconds
    time_s = raw_time * 1e-6

    os.makedirs(args.output_dir, exist_ok=True)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    pad_x = 10.0
    pad_y = 10.0
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    print(f"Loaded keys: time={keys[0]}, x={keys[1]}, y={keys[2]}")
    print(f"Saving outputs to: {args.output_dir}")
    print(f"Timestamp scale: microseconds -> seconds for computation (x1e-6)")
    print(f"Timestamp display: milliseconds (x1e-3)")
    print(f"Number of bins: {args.num_bins}")
    print(f"Events used per bin (max): {args.analysis_events_per_bin}")
    print(f"Minimum points for PCA: {args.min_points_for_pca}")
    print(f"Center estimation method: {args.center_method}")

    if args.origin_x is None or args.origin_y is None:
        estimated_cx, estimated_cy = estimate_global_center(x, y, method=args.center_method)
        if args.origin_x is None:
            args.origin_x = estimated_cx
        if args.origin_y is None:
            args.origin_y = estimated_cy

    print(f"Global origin / center: ({args.origin_x:.3f}, {args.origin_y:.3f})")

    first_ts_s = float(time_s[0])
    last_ts_s = float(time_s[-1])
    if last_ts_s <= first_ts_s:
        raise ValueError("Invalid timestamp range: last_ts must be greater than first_ts.")

    run_pca_tracking(raw_time, x, y, args, xlim, ylim)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)