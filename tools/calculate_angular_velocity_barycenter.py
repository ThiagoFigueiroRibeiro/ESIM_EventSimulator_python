
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
# Geometry / angles
# -----------------------------
def centroid_2d(x, y):
    pts = np.column_stack([x, y])
    return pts.mean(axis=0)


def estimate_global_center(x, y, method="mean"):
    method = method.lower()
    if method == "mean":
        return float(np.mean(x)), float(np.mean(y))
    elif method == "median":
        return float(np.median(x)), float(np.median(y))
    else:
        raise ValueError(f"Unknown center method: {method}")


def angle_from_origin_signed(px, py, ox=320.0, oy=320.0):
    """
    Signed angle in image coordinates, matching code 1's convention:
      angle = atan2(oy - py, px - ox)

    This returns a value in (-pi, pi].
    """
    return float(np.arctan2(oy - py, px - ox))


def wrap_to_pi(theta):
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def unwrap_continuous_angle(theta_obs, prev_theta=None):
    if prev_theta is None or not np.isfinite(prev_theta):
        return float(theta_obs)
    return float(prev_theta + wrap_to_pi(theta_obs - prev_theta))


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
# Oscillatory fit
# -----------------------------
def estimate_period_guess_fft(t_ms, y):
    """
    Rough period estimate using detrended FFT on interpolated uniform samples.
    Returns a period in ms, or np.nan if it cannot estimate one.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t_ms) < 6:
        return np.nan

    order = np.argsort(t_ms)
    t_ms = t_ms[order]
    y = y[order]

    span = float(t_ms[-1] - t_ms[0])
    if span <= 0:
        return np.nan

    try:
        p = np.polyfit(t_ms, y, 1)
        y_detrended = y - np.polyval(p, t_ms)
    except Exception:
        y_detrended = y - np.mean(y)

    n_grid = int(min(4096, max(256, 8 * len(t_ms))))
    grid = np.linspace(t_ms[0], t_ms[-1], n_grid)
    grid_y = np.interp(grid, t_ms, y_detrended)
    grid_y -= np.mean(grid_y)

    dt = grid[1] - grid[0]
    if dt <= 0:
        return np.nan

    freqs = np.fft.rfftfreq(n_grid, d=dt)
    spec = np.abs(np.fft.rfft(grid_y))

    if len(freqs) <= 1:
        return np.nan

    idx = np.argmax(spec[1:]) + 1
    f = float(freqs[idx])

    if not np.isfinite(f) or f <= 0:
        return np.nan

    return float(1.0 / f)


def fit_oscillatory_motion(t_ms, theta_rad):
    """
    Fit:
      theta(x) = theta0 + v0*(x - x_ref) + A*sin(2*pi*(x - x_ref)/T + phi)

    We fit an equivalent linearized form for each candidate period T:
      theta(x) = a + b*t + c*sin(wt) + d*cos(wt)

    Then convert c,d to A,phi.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    theta_rad = np.asarray(theta_rad, dtype=float)

    mask = np.isfinite(t_ms) & np.isfinite(theta_rad)
    t_ms = t_ms[mask]
    theta_rad = theta_rad[mask]

    if len(t_ms) < 6:
        return None

    order = np.argsort(t_ms)
    t_ms = t_ms[order]
    theta_rad = theta_rad[order]

    x_ref = float(t_ms[0])
    x = t_ms - x_ref
    span = float(x[-1] - x[0])
    if span <= 0:
        return None

    T_min = max(span / 100.0, 1e-6)
    T_max = max(span * 10.0, T_min * 10.0)

    candidates = list(np.logspace(np.log10(T_min), np.log10(T_max), 600))

    T_guess = estimate_period_guess_fft(t_ms, theta_rad)
    if np.isfinite(T_guess) and T_guess > 0:
        local = np.linspace(0.5 * T_guess, 1.5 * T_guess, 250)
        local = local[np.isfinite(local) & (local > 0)]
        candidates.extend(local.tolist())

    candidates = np.array(sorted(set(float(c) for c in candidates if np.isfinite(c) and c > 0)))
    if len(candidates) == 0:
        return None

    best = None

    for T in candidates:
        w = 2.0 * np.pi / T
        s = np.sin(w * x)
        c = np.cos(w * x)
        X = np.column_stack([np.ones_like(x), x, s, c])

        try:
            beta, *_ = np.linalg.lstsq(X, theta_rad, rcond=None)
        except Exception:
            continue

        resid = theta_rad - X @ beta
        rss = float(np.sum(resid ** 2))

        if best is None or rss < best["rss"]:
            best = {
                "rss": rss,
                "T": float(T),
                "beta": beta,
            }

    if best is None:
        return None

    theta0, v0, c_sin, d_cos = best["beta"]
    T = best["T"]
    w = 2.0 * np.pi / T

    A = float(np.hypot(c_sin, d_cos))
    phi = float(np.arctan2(d_cos, c_sin))

    def theta_fit_func(x_ms_in):
        x_ms_in = np.asarray(x_ms_in, dtype=float)
        xr = x_ms_in - x_ref
        return theta0 + v0 * xr + A * np.sin(w * xr + phi)

    def omega_fit_func(x_ms_in):
        x_ms_in = np.asarray(x_ms_in, dtype=float)
        xr = x_ms_in - x_ref
        return v0 + A * w * np.cos(w * xr + phi)

    eq_theta = (
        f"θ(x)=θ0 + v0(x-x0) + A sin(2π(x-x0)/T + φ)  "
        f"[x0={x_ref:.4g}, θ0={theta0:.4g}, v0={v0:.4g}, A={A:.4g}, T={T:.4g}, φ={phi:.4g}]"
    )
    eq_omega = (
        f"v(x)=v0 + A(2π/T) cos(2π(x-x0)/T + φ)  "
        f"[x0={x_ref:.4g}, v0={v0:.4g}, A={A:.4g}, T={T:.4g}, φ={phi:.4g}]"
    )

    return {
        "x_ref": x_ref,
        "theta0": float(theta0),
        "v0": float(v0),
        "A": float(A),
        "T": float(T),
        "phi": float(phi),
        "rss": best["rss"],
        "theta_fit_func": theta_fit_func,
        "omega_fit_func": omega_fit_func,
        "eq_theta": eq_theta,
        "eq_omega": eq_omega,
    }


# -----------------------------
# Plotting
# -----------------------------
def save_centroid_frame(
    out_path,
    xs,
    ys,
    centroid_x,
    centroid_y,
    prev_centroid_x,
    prev_centroid_y,
    origin_x,
    origin_y,
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

    # Global origin
    ax.scatter(
        [origin_x], [origin_y],
        s=80, c="red", marker="x", linewidths=2.5, label="origin"
    )

    # Previous centroid and motion arrow
    have_current = centroid_x is not None and centroid_y is not None
    have_prev = prev_centroid_x is not None and prev_centroid_y is not None

    if have_prev:
        ax.scatter([prev_centroid_x], [prev_centroid_y], s=55, c="orange", marker="o", label="previous centroid")

    if have_current:
        ax.scatter([centroid_x], [centroid_y], s=70, c="blue", marker="o", label="centroid")

        # Arrow from origin to current centroid
        ax.annotate(
            "",
            xy=(centroid_x, centroid_y),
            xytext=(origin_x, origin_y),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2)
        )

        # Arrow showing centroid movement from previous timestep
        if have_prev:
            draw_single_arrow(
                ax,
                (prev_centroid_x, prev_centroid_y),
                (centroid_x, centroid_y),
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
                f"No valid angle",
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
            f"No valid centroid",
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
def run_centroid_tracking(raw_time, x, y, args, xlim, ylim):
    global_cx, global_cy = estimate_global_center(x, y, method=args.center_method)

    if args.origin_x is None:
        args.origin_x = global_cx
    if args.origin_y is None:
        args.origin_y = global_cy

    tracking_frames_dir = (
        os.path.join(args.frames_base_dir, "tracking_frames_centroid")
        if args.frames_base_dir else "tracking_frames_centroid"
    )
    os.makedirs(tracking_frames_dir, exist_ok=True)
    print(f"Saving tracking frames to: {tracking_frames_dir}")

    n_events_total = len(x)
    if args.num_bins <= 0:
        raise ValueError("--num_bins must be > 0")

    bin_edges = np.linspace(0, n_events_total, args.num_bins + 1, dtype=int)
    n_bins = args.num_bins

    bin_ids = np.arange(n_bins, dtype=int)
    start_event_idx = np.full(n_bins, -1, dtype=int)
    end_event_idx = np.full(n_bins, -1, dtype=int)
    n_events_per_bin = np.zeros(n_bins, dtype=int)

    window_t_ms = np.full(n_bins, np.nan, dtype=float)
    window_t_start_ms = np.full(n_bins, np.nan, dtype=float)
    window_t_end_ms = np.full(n_bins, np.nan, dtype=float)

    centroid_xs = np.full(n_bins, np.nan, dtype=float)
    centroid_ys = np.full(n_bins, np.nan, dtype=float)

    angle_obs = np.full(n_bins, np.nan, dtype=float)
    angle_unwrapped = np.full(n_bins, np.nan, dtype=float)

    omega_rad_s = np.full(n_bins, np.nan, dtype=float)
    omega_rad_ms = np.full(n_bins, np.nan, dtype=float)

    valid_measurement = np.zeros(n_bins, dtype=int)
    propagated_only = np.zeros(n_bins, dtype=int)

    prev_theta = None
    prev_omega = None
    prev_time_s = None
    prev_centroid_x = None
    prev_centroid_y = None

    frame_data = []

    for i in range(n_bins):
        bin_s = int(bin_edges[i])
        bin_e = int(bin_edges[i + 1])

        # Equal-count bins, but use only the first N events inside each bin
        e_use = min(bin_s + args.analysis_events_per_bin, bin_e)

        start_event_idx[i] = bin_s
        end_event_idx[i] = e_use

        xs = x[bin_s:e_use]
        ys = y[bin_s:e_use]
        ts = raw_time[bin_s:e_use] * 1e-6  # microseconds -> seconds

        n_events = len(xs)
        n_events_per_bin[i] = n_events

        centroid_valid = False
        cx = None
        cy = None
        t_center_s = np.nan

        if n_events > 0:
            t_center_s = float(np.mean(ts))
            window_t_ms[i] = t_center_s * 1e3
            window_t_start_ms[i] = float(ts[0] * 1e3)
            window_t_end_ms[i] = float(ts[-1] * 1e3)

            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            centroid_xs[i] = cx
            centroid_ys[i] = cy
            centroid_valid = True
            valid_measurement[i] = 1

            theta_o = angle_from_origin_signed(cx, cy, args.origin_x, args.origin_y)

            if prev_theta is None or not np.isfinite(prev_theta):
                theta_u = float(theta_o)
                omega_s = np.nan
            else:
                dt = t_center_s - prev_time_s if prev_time_s is not None and np.isfinite(prev_time_s) else np.nan
                theta_u = unwrap_continuous_angle(theta_o, prev_theta)
                if np.isfinite(dt) and dt > 0:
                    omega_s = (theta_u - prev_theta) / dt
                else:
                    omega_s = np.nan

            angle_obs[i] = theta_o
            angle_unwrapped[i] = theta_u
            omega_rad_s[i] = omega_s
            omega_rad_ms[i] = omega_s * 1e-3 if np.isfinite(omega_s) else np.nan

            prev_theta = theta_u
            prev_omega = omega_s if np.isfinite(omega_s) else prev_omega
            prev_time_s = t_center_s
            prev_centroid_x = cx
            prev_centroid_y = cy
        else:
            # Missing bin
            if (
                args.propagate_missing
                and prev_theta is not None
                and np.isfinite(prev_theta)
                and prev_time_s is not None
                and np.isfinite(prev_time_s)
            ):
                # Only meaningful if we can estimate dt; here no events -> no reliable timestamp
                # so we leave it as missing unless you later want bin-center-time propagation.
                propagated_only[i] = 1
            theta_u = np.nan
            omega_s = np.nan

        frame_data.append({
            "bin_id": i,
            "xs": xs,
            "ys": ys,
            "centroid_x": cx if centroid_valid else None,
            "centroid_y": cy if centroid_valid else None,
            "prev_centroid_x": prev_centroid_x if centroid_valid else None,
            "prev_centroid_y": prev_centroid_y if centroid_valid else None,
            "angle_obs": angle_obs[i],
            "angle_unwrapped": angle_unwrapped[i],
            "omega_ms": omega_rad_ms[i],
            "t_start_ms": window_t_start_ms[i],
            "t_end_ms": window_t_end_ms[i],
            "t_center_ms": window_t_ms[i],
            "n_events": n_events,
            "valid": centroid_valid,
        })

    # Generate tracking frames
    for fd in tqdm(frame_data, desc="Generating tracking frames", unit="frame"):
        i = fd["bin_id"]
        out_path = os.path.join(
            tracking_frames_dir,
            f"{args.frame_prefix}_{i:05d}.{args.image_ext}"
        )

        save_centroid_frame(
            out_path=out_path,
            xs=fd["xs"],
            ys=fd["ys"],
            centroid_x=fd["centroid_x"],
            centroid_y=fd["centroid_y"],
            prev_centroid_x=fd["prev_centroid_x"],
            prev_centroid_y=fd["prev_centroid_y"],
            origin_x=args.origin_x,
            origin_y=args.origin_y,
            theta_obs=fd["angle_obs"] if np.isfinite(fd["angle_obs"]) else np.nan,
            theta_unwrapped=fd["angle_unwrapped"] if np.isfinite(fd["angle_unwrapped"]) else np.nan,
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

    valid_angle = np.isfinite(angle_unwrapped)
    valid_vel = np.isfinite(omega_rad_ms)

    if not np.any(valid_angle):
        print("No valid tracking bins. Nothing to plot.")
        return

    # -----------------------------
    # Fit oscillatory model if requested
    # -----------------------------
    fit_result = None
    if args.angular_fit:
        fit_result = fit_oscillatory_motion(window_t_ms[valid_angle], angle_unwrapped[valid_angle])
        if fit_result is not None:
            print()
            print("Angular fit")
            print("-----------")
            print(fit_result["eq_theta"])
            print(fit_result["eq_omega"])
        else:
            print("WARNING: --angular_fit was requested but the fit could not be computed.")

    # -----------------------------
    # Stats
    # -----------------------------
    print()
    print("Angular position stats")
    print("----------------------")
    print(f"Valid bins with angle:     {int(np.sum(valid_angle))} / {n_bins}")
    print(f"Valid centroid measurements: {int(np.sum(valid_measurement))} / {n_bins}")
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
        centroid_xs,
        centroid_ys,
        angle_obs,
        angle_unwrapped,
        omega_rad_s,
        omega_rad_ms,
    ])

    csv_header = (
        "bin,start_event_idx,end_event_idx,"
        "t_start_ms,t_end_ms,t_center_ms,"
        "n_events,valid_measurement,propagated_only,"
        "centroid_x,centroid_y,"
        "angle_obs_rad,angle_unwrapped_rad,"
        "angular_velocity_rad_s,angular_velocity_rad_ms"
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
            "%.10e", "%.10e"
        ]
    )
    print(f"Saved CSV to: {csv_path}")

    # Precompute fit curves if needed
    if fit_result is not None:
        valid_t = window_t_ms[valid_angle]
        fit_t_dense = np.linspace(float(np.min(valid_t)), float(np.max(valid_t)), 1000)
        fit_theta_dense = fit_result["theta_fit_func"](fit_t_dense)
        fit_omega_dense = fit_result["omega_fit_func"](fit_t_dense)
        eq_theta = fit_result["eq_theta"]
        eq_omega = fit_result["eq_omega"]
    else:
        fit_t_dense = None
        fit_theta_dense = None
        fit_omega_dense = None
        eq_theta = None
        eq_omega = None

    # -----------------------------
    # Plot angular position over time
    # -----------------------------
    pos_plot_path = os.path.join(args.output_dir, f"angular_position_over_time.{args.image_ext}")
    fig, ax = plt.subplots(figsize=(10, 5))

    if fit_result is not None:
        ax.plot(
            fit_t_dense,
            fit_theta_dense,
            color="green",
            linewidth=2.5,
            alpha=0.95,
            label="oscillatory fit",
            zorder=1
        )

    if np.any(valid_angle):
        ax.plot(
            window_t_ms[valid_angle],
            angle_unwrapped[valid_angle],
            marker="o",
            linewidth=1.5,
            markersize=3,
            color="purple",
            label="unwrapped centroid angle",
            zorder=3
        )

    meas_idx = valid_measurement.astype(bool) & np.isfinite(angle_unwrapped)
    if np.any(meas_idx):
        ax.scatter(
            window_t_ms[meas_idx],
            angle_unwrapped[meas_idx],
            s=20,
            color="orange",
            alpha=0.8,
            label="measured bins",
            zorder=4
        )

    prop_idx = propagated_only.astype(bool) & np.isfinite(angle_unwrapped)
    if np.any(prop_idx):
        ax.scatter(
            window_t_ms[prop_idx],
            angle_unwrapped[prop_idx],
            s=20,
            color="gray",
            alpha=0.8,
            label="propagated bins",
            zorder=4
        )

    if fit_result is not None:
        ax.set_title(f"Angular position over time\n{eq_theta}", fontsize=12.5, pad=16)
    else:
        ax.set_title("Angular position over time", fontsize=14, pad=16)

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("angular position [rad]")
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

    if fit_result is not None:
        ax.plot(
            fit_t_dense,
            fit_omega_dense,
            color="green",
            linewidth=2.5,
            alpha=0.95,
            label="oscillatory fit",
            zorder=1
        )

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
            label="angular velocity",
            zorder=3
        )

        ax.axhline(
            mean_omega,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"mean ω = {mean_omega:.3e} rad/ms ({mean_omega_deg:.3e} deg/ms)",
            zorder=2
        )

        ax.fill_between(
            t_valid_ms,
            mean_omega - mae_omega,
            mean_omega + mae_omega,
            color="red",
            alpha=0.12,
            label=f"±MAE = {mae_omega:.3e} rad/ms",
            zorder=0
        )
        ax.fill_between(
            t_valid_ms,
            mean_omega - rmse_omega,
            mean_omega + rmse_omega,
            color="orange",
            alpha=0.12,
            label=f"±RMSE = {rmse_omega:.3e} rad/ms",
            zorder=0
        )

    if fit_result is not None:
        ax.set_title(f"Angular velocity over time\n{eq_omega}", fontsize=12.5, pad=16)
    else:
        ax.set_title("Angular velocity over time", fontsize=14, pad=16)

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("angular velocity [rad/ms]")
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
        description="Centroid-based continuous angle tracker using equal-count bins and the first N events per bin."
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
        help="Use only the first N events of each bin for centroid/angle estimation."
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
        help="Base directory for tracking_frames_centroid. Empty means current directory."
    )

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--image_ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--frame_prefix", type=str, default="tracking_frame")

    parser.add_argument("--plot_events", action="store_true", default=True, help="Plot events in the frame.")
    parser.add_argument("--no_plot_events", dest="plot_events", action="store_false", help="Disable event plotting.")
    parser.add_argument("--events_marker_size", type=float, default=8.0)

    parser.add_argument("--arrow_lw", type=float, default=3.0, help="Arrow line width.")
    parser.add_argument("--arrow_mutation_scale", type=float, default=18.0, help="Arrowhead size.")

    parser.add_argument("--show", action="store_true", help="Show figures while generating.")

    # Oscillatory fit
    parser.add_argument(
        "--angular_fit",
        action="store_true",
        default=False,
        help="Fit a constant-drift + oscillatory model and overlay fitted curves."
    )

    # Global center / origin
    parser.add_argument(
        "--center_method",
        type=str,
        default="mean",
        choices=["mean", "median"],
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
        help="If a bin lacks enough events, propagate state when possible."
    )
    parser.add_argument(
        "--no_propagate_missing",
        dest="propagate_missing",
        action="store_false",
        help="Disable propagation for bins that lack enough events."
    )

    args = parser.parse_args()

    if args.num_bins <= 0:
        raise ValueError("--num_bins must be > 0")
    if args.analysis_events_per_bin <= 0:
        raise ValueError("--analysis_events_per_bin must be > 0")

    raw_time, x, y, keys = load_events(args.npz)

    # timestamps are assumed microseconds
    time_s = raw_time * 1e-6
    time_ms = raw_time * 1e-3

    os.makedirs(args.output_dir, exist_ok=True)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    xlim_min = min(x_min, args.origin_x if args.origin_x is not None else x_min)
    xlim_max = max(x_max, args.origin_x if args.origin_x is not None else x_max)
    ylim_min = min(y_min, args.origin_y if args.origin_y is not None else y_min)
    ylim_max = max(y_max, args.origin_y if args.origin_y is not None else y_max)

    pad_x = 10.0
    pad_y = 10.0
    xlim = (xlim_min - pad_x, xlim_max + pad_x)
    ylim = (ylim_min - pad_y, ylim_max + pad_y)

    print(f"Loaded keys: time={keys[0]}, x={keys[1]}, y={keys[2]}")
    print(f"Saving outputs to: {args.output_dir}")
    print(f"Timestamp scale: microseconds -> seconds for computation (x1e-6)")
    print(f"Timestamp display: milliseconds (x1e-3)")
    print(f"Number of bins: {args.num_bins}")
    print(f"Events used per bin (max): {args.analysis_events_per_bin}")
    print(f"Center estimation method: {args.center_method}")
    print(f"Angular fit enabled: {args.angular_fit}")

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

    run_centroid_tracking(raw_time, x, y, args, xlim, ylim)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)