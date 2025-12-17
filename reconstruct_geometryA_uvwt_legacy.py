#!/usr/bin/env python3

import numpy as np
import math
from numpy.linalg import svd, norm
import testing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# CONSTANTS
# ============================================================

REF_X = -138.9971
REF_Y = 98.25
STRIP_PITCH = 1.5
SAMPLING_FREQUENCY = 25.0  # MHz
DRIFT_VELOCITY = 6.46      # mm/us
F_TBIN_TO_MM = DRIFT_VELOCITY / SAMPLING_FREQUENCY
T_OFFSET_CONST = 256.0
TRIGGER_DELAY = 5.0

PHIS = {
    "U": 0.0,
    "V": np.deg2rad(30.0),
    "W": np.deg2rad(-30.0)
}

# ============================================================
# FITTING: centroid-based per-time-bin + weighted LS
# ============================================================

def fit_2d_line_from_hist(hist, t_threshold_frac=0.01, min_bins=8, fallback_percentile=90.0):
    """
    Robust ridge finder: compute weighted centroid per time bin, then fit a line
    strip_index = a * t + b using weighted least squares.

    Parameters
    ----------
    hist : 2D ndarray
        time x strip index histogram
    t_threshold_frac : float
        fraction of peak row-sum to treat a time row as active
    min_bins : int
        minimal number of time bins required for centroid fit; otherwise fallback
    fallback_percentile : float
        if centroid approach fails, fallback to global-percentile weighted LS.

    Returns
    -------
    a, b : floats
        slope and intercept for strip = a * t + b
    """
    H, W = hist.shape
    row_sum = np.sum(hist, axis=1)
    peak = float(np.max(row_sum)) if np.max(row_sum) > 0 else 1.0
    thr = t_threshold_frac * peak

    ts = []
    cs = []
    ws = []

    strip_indices = np.arange(W, dtype=float)
    for t in range(H):
        S = row_sum[t]
        if S > thr:
            row = hist[t, :].astype(float)
            # centroid: weighted mean of strip indices
            c = float((strip_indices * row).sum() / (row.sum() + 1e-12))
            ts.append(float(t))
            cs.append(c)
            ws.append(float(row.sum()))

    if len(ts) >= min_bins:
        ts = np.array(ts, dtype=float)
        cs = np.array(cs, dtype=float)
        ws = np.array(ws, dtype=float)
        w = np.sqrt(np.maximum(ws, 1e-12))  # use sqrt of charge as weights
        A = np.vstack([ts, np.ones_like(ts)]).T
        Aw = A * w[:, None]
        bw = cs * w
        p, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        a, b = float(p[0]), float(p[1])
        return a, b

    # Fallback: previous global percentile approach (less robust)
    T, C = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    Tf = T.ravel()
    Cf = C.ravel()
    If = hist.ravel()

    thrg = np.percentile(If, fallback_percentile)
    sel = If >= thrg
    if sel.sum() < 10:
        sel = If > 0
    weights = np.sqrt(np.maximum(If[sel], 1e-12))
    A = np.vstack([Tf[sel], np.ones(sel.sum())]).T
    Aw = A * weights[:, None]
    bw = Cf[sel] * weights

    p, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    a, b = float(p[0]), float(p[1])
    return a, b

# ============================================================
# INVERSE MAPPING (UVWT -> XYZ) -- detector frame
# ============================================================

def invert_uvwt_point(u_bin, v_bin, w_bin, t_bin, phis=PHIS):
    """
    Invert fractional strip indices (u_bin, v_bin, w_bin) and time bin t_bin
    to detector-frame coordinates (x_mm, y_mm, z_mm).
    """
    # y from U-plane (forward: u = -(y - 99.75)/pitch)
    y_mm = 99.75 - STRIP_PITCH * float(u_bin)

    # V/W plane positions (mm)
    U_mm = STRIP_PITCH * float(v_bin) - 98.75
    W_mm = STRIP_PITCH * float(w_bin) - 98.75

    Vphi = float(phis["V"])
    Wphi = float(phis["W"])
    cosV = math.cos(Vphi)
    sinV = math.sin(Vphi)
    cosW = math.cos(-Wphi)
    sinW = math.sin(-Wphi)

    rhs_v = U_mm + (y_mm - REF_Y) * sinV
    rhs_w = W_mm + (y_mm - REF_Y) * sinW

    denom = cosV * cosV + cosW * cosW
    if abs(denom) < 1e-12:
        denom = 1e-12  # guard

    x_minus_ref = (cosV * rhs_v + cosW * rhs_w) / denom
    x_mm = REF_X + x_minus_ref

    z_mm = (float(t_bin) - T_OFFSET_CONST - TRIGGER_DELAY) * F_TBIN_TO_MM

    return x_mm, y_mm, z_mm

# ============================================================
# RECONSTRUCTION: sample only active time window, clamp indices
# ============================================================

def reconstruct_from_histograms_notebook(raw, n_samples=400, active_frac_thresh=0.01, margin_frac=0.02):
    """
    Reconstruct 3D point cloud and PCA track from histograms.

    Returns detector-frame outputs and diagnostic 't_range'.
    """
    histU, histV, histW = raw[0][0]  # each is H x W (time x strips)
    histU, histV, histW = raw[0][0]  # each is H x W (time x strips)
    H, W = histU.shape

    # Fit lines with centroid-based fitter
    a_u, b_u = fit_2d_line_from_hist(histU)
    a_v, b_v = fit_2d_line_from_hist(histV)
    a_w, b_w = fit_2d_line_from_hist(histW)

    # Determine active time window from summed activity across planes
    sumU = np.sum(histU, axis=1)
    sumV = np.sum(histV, axis=1)
    sumW = np.sum(histW, axis=1)
    sum_all = sumU + sumV + sumW
    peak = float(np.max(sum_all)) if np.max(sum_all) > 0 else 1.0
    thr = active_frac_thresh * peak
    active_idx = np.where(sum_all > thr)[0]
    if active_idx.size == 0:
        t_min, t_max = 0, H - 1
    else:
        t_min, t_max = int(active_idx[0]), int(active_idx[-1])

    # Add tiny margin to ensure we include boundary bins
    margin = max(1, int(margin_frac * (t_max - t_min + 1)))
    t_min = max(0, t_min - margin)
    t_max = min(H - 1, t_max + margin)

    # Sample only inside active window (avoids extrapolation)
    t_vals = np.linspace(t_min, t_max, n_samples)

    # Compute fractional strip indices from fits and clamp to [0, W-1]
    u_bins = np.clip(a_u * t_vals + b_u, 0.0, float(W - 1))
    v_bins = np.clip(a_v * t_vals + b_v, 0.0, float(W - 1))
    w_bins = np.clip(a_w * t_vals + b_w, 0.0, float(W - 1))

    # Invert to mm coordinates
    xs = np.empty(n_samples)
    ys = np.empty(n_samples)
    zs = np.empty(n_samples)
    for i in range(n_samples):
        xs[i], ys[i], zs[i] = invert_uvwt_point(u_bins[i], v_bins[i], w_bins[i], t_vals[i])

    pts = np.column_stack((xs, ys, zs))

    # PCA to obtain direction
    center = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd((pts - center).T, full_matrices=False)
    direction = U[:, 0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    proj = (pts - center) @ direction
    ep0 = center + direction * proj.min()
    ep1 = center + direction * proj.max()

    return {
        "center": center,
        "direction": direction,
        "ep0_mm": ep1,
        "ep1_mm": ep0,
        "pts": pts,
        "fits": ((a_u, b_u), (a_v, b_v), (a_w, b_w)),
        "t_range": (int(t_min), int(t_max)),
    }

# ============================================================
# RAW ERROR (detector frame)
# ============================================================

def compute_raw_errors(detector_result, raw):
    """
    Compute raw endpoint errors in detector frame (this is the meaningful performance metric).
    """
    gt0 = np.array(raw[0][1], float)
    gt1 = np.array(raw[0][2], float)
    ep0 = np.array(detector_result["ep0_mm"], float)
    ep1 = np.array(detector_result["ep1_mm"], float)

    err0 = float(min(norm(ep0 - gt0), norm(ep0 - gt1)))
    err1 = float(min(norm(ep1 - gt0), norm(ep1 - gt1)))

    return {
        "ep0_raw": ep0,
        "ep1_raw": ep1,
        "err0_raw": err0,
        "err1_raw": err1,
        "gt0": gt0,
        "gt1": gt1
    }

# ============================================================
# PLOTTING UTILITIES
# ============================================================

def plot_histogram_with_fit(hist, a, b, t_range=None, title="histogram + fit",
                            cmap="viridis", vperc=(1, 99), figsize=(6, 5), dpi=150,
                            show_centroids=True):
    """
    Improved histogram plot:
     - shows centroids used for fitting (white dots)
     - overlays fit only on t_range (if provided)
     - percentile-based vmin/vmax for good contrast
    """
    H, W = hist.shape
    vmin = np.percentile(hist, vperc[0])
    vmax = np.percentile(hist, vperc[1])

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    im = ax.imshow(hist, aspect='auto', origin='lower',
                   interpolation='nearest', vmin=vmin, vmax=vmax,
                   extent=[0, W - 1, 0, H - 1], cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Intensity")

    # t domain for plotting the fit
    if t_range is None:
        t_plot = np.linspace(0, H - 1, 600)
    else:
        t_min, t_max = t_range
        t_plot = np.linspace(t_min, t_max, 600)

    strip_plot = a * t_plot + b
    strip_plot = np.clip(strip_plot, 0.0, float(W - 1.0))
    ax.plot(strip_plot, t_plot, color='red', linewidth=2, label='line fit')

    # overlay centroids used for fit
    if show_centroids:
        row_sum = np.sum(hist, axis=1)
        threshold = 0.01 * max(row_sum.max(), 1.0)
        t_bins = []
        centroid_x = []
        for t in range(H):
            S = row_sum[t]
            if S > threshold:
                c = float((np.arange(W) * hist[t, :]).sum() / (S + 1e-12))
                t_bins.append(t)
                centroid_x.append(c)
        if len(t_bins) > 0:
            ax.scatter(centroid_x, t_bins, s=10, c='white', edgecolors='black', linewidths=0.2, zorder=5, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Strip index")
    ax.set_ylabel("Time bin")
    ax.set_xlim(0, W - 1)
    ax.set_ylim(0, H - 1)
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig, ax

def set_axes_equal(ax):
    """
    Make 3D axes have equal scale (prevents distortion).
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    if plot_radius == 0:
        plot_radius = 1.0
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_3d_track(points, ep0, ep1, figsize=(8, 6), dpi=150, cmap='plasma'):
    """
    3D scatter of reconstructed points with PCA-based segment overlay.
    Color coded by sampling order/time.
    """
    N = points.shape[0]
    times = np.linspace(0.0, 1.0, N)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=times, cmap=cmap, s=10, alpha=0.9, label='Reconstructed points')

    # line segment
    ax.plot([ep0[0], ep1[0]], [ep0[1], ep1[1]], [ep0[2], ep1[2]], 'r-', linewidth=3, label='Track segment')

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("3D Reconstructed Track (Detector Frame)")
    ax.legend(loc='upper left')
    set_axes_equal(ax)
    plt.tight_layout()
    # colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("sampling order")
    return fig, ax

# ============================================================
# MAIN REPORTING/ENTRY
# ============================================================

def main(n_samples=600, show_plots=True):
    raw = testing.getTestData("middle")  # 'middle' canonical set in testing.py
    recon = reconstruct_from_histograms_notebook(raw, n_samples=n_samples)
    metrics = compute_raw_errors(recon, raw)

    print("\n=== RAW RECONSTRUCTION ===\n")
    print("Detector-frame (raw) endpoints:")
    print(" ep0:", metrics["ep0_raw"])
    print(" ep1:", metrics["ep1_raw"])
    print("\nGround-truth endpoints (for reference):")
    print(" gt0:", metrics["gt0"])
    print(" gt1:", metrics["gt1"])
    print("\nRAW endpoint errors (mm):")
    print(f" ||ep0 - gt0|| = {metrics['err0_raw']:.6f}")
    print(f" ||ep1 - gt1|| = {metrics['err1_raw']:.6f}\n")

    # Optional visuals
    if show_plots:
        histU, histV, histW = raw[0][0]
        (a_u, b_u), (a_v, b_v), (a_w, b_w) = recon["fits"]
        t_range = recon.get("t_range", None)

        # histogram fits
        plot_histogram_with_fit(histU, a_u, b_u, t_range=t_range, title="U-plane histogram + line fit")
        plot_histogram_with_fit(histV, a_v, b_v, t_range=t_range, title="V-plane histogram + line fit")
        plot_histogram_with_fit(histW, a_w, b_w, t_range=t_range, title="W-plane histogram + line fit")

        # 3D track
        plot_3d_track(recon["pts"], recon["ep0_mm"], recon["ep1_mm"])

        plt.show()

    return metrics

if __name__ == "__main__":
    main(show_plots = False)

