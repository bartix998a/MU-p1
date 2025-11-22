# reconstruct_geometryA_uvwt.py
"""
Approach A — geometry A (3 strip planes rotated about Z: 0°, +30°, -30°).
Direct inversion of strip projections -> XYZ -> PCA line fit.

Usage:
    python reconstruct_geometryA_uvwt.py
"""

import numpy as np
from numpy.linalg import lstsq, svd

# detector / sim constants (same as in testing.py)
STRIP_PITCH = 1.5  # mm per strip index
SAMPLING_FREQUENCY = 25.0  # MHz
DRIFT_VELOCITY = 6.46  # mm / us
MM_PER_TBIN = DRIFT_VELOCITY / SAMPLING_FREQUENCY
T_OFFSET_CONST = 256.0
TRIGGER_DELAY = 5.0

# plane angles (radians) around Z axis
PHIS = {
    "U": 0.0,
    "V": np.deg2rad(+30.0),
    "W": np.deg2rad(-30.0)
}

# helper: least squares PCA line fit
def fit_line_pca(points):
    pts = np.asarray(points).reshape(-1, 3)
    center = pts.mean(axis=0)
    U, S, Vt = svd((pts - center).T, full_matrices=False)
    direction = U[:, 0]
    direction /= (np.linalg.norm(direction) + 1e-12)
    return center, direction

# fit robust 2D line (t -> strip_index) from histogram
def fit_2d_line_from_hist(hist, keep_percentile=90.0):
    H, W = hist.shape
    T, C = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    Tf = T.ravel(); Cf = C.ravel(); If = hist.ravel()
    thr = np.percentile(If, keep_percentile)
    sel = If >= thr
    if sel.sum() < 50:
        sel = If > 0
    Wv = np.sqrt(np.maximum(If[sel], 1e-12))
    A = np.vstack([Tf[sel], np.ones(sel.sum())]).T
    Aw = A * Wv[:, None]
    bw = Cf[sel] * Wv
    p, *_ = lstsq(Aw, bw, rcond=None)
    return float(p[0]), float(p[1])

# inverse mapping: from three strip coordinates (s_mm for each plane) to (x,y)
# s_mm_i = x*cos(phi_i) + y*sin(phi_i)   (this is projection on axis at phi)
# Solve linear system M [x; y] = s_vector
def solve_xy_from_s(s_mm_list, phis_dict):
    phis = [float(phis_dict["U"]), float(phis_dict["V"]), float(phis_dict["W"])]
    M = np.vstack([[np.cos(p), np.sin(p)] for p in phis])  # shape (3,2)
    s = np.asarray(s_mm_list).reshape(-1)
    # least squares (3x2) -> (x,y)
    sol, *_ = lstsq(M, s, rcond=None)
    return float(sol[0]), float(sol[1])

# convert strip-bin (index) to s_mm coordinate: s_mm = (strip_index - center_bin) * strip_pitch
# We'll set center_bin = hist.shape[1] / 2 so that bin indexes are centered
def strip_bin_to_mm(strip_bin, center_bin, strip_pitch=STRIP_PITCH):
    return (np.asarray(strip_bin, float) - float(center_bin)) * strip_pitch

def reconstruct_from_histograms(hist_list, phis=PHIS, n_samples=400, verbose=True):
    # accept hist_list or tuple returned by getTestData
    if isinstance(hist_list, tuple) or (isinstance(hist_list, list) and len(hist_list) > 0 and not hasattr(hist_list[0], "shape")):
        # possibly the full getTestData tuple, choose first element
        hist_list = hist_list[0]

    if not isinstance(hist_list, (list, tuple)) or len(hist_list) != 3:
        raise ValueError("hist_list must be list of 3 histograms")

    hU, hV, hW = [np.asarray(h, float) for h in hist_list]
    H, W = hU.shape

    # 1) Fit t->strip line in each histogram
    a_u, b_u = fit_2d_line_from_hist(hU)
    a_v, b_v = fit_2d_line_from_hist(hV)
    a_w, b_w = fit_2d_line_from_hist(hW)

    # 2) Build sampled t values (bins) and predicted strip indices
    t_vals = np.linspace(0, H - 1, n_samples)
    u_bins = a_u * t_vals + b_u
    v_bins = a_v * t_vals + b_v
    w_bins = a_w * t_vals + b_w

    # 3) Convert strip-bin -> mm coordinate along each projection axis
    center_bin = W / 2.0
    s_u_mm = strip_bin_to_mm(u_bins, center_bin)
    s_v_mm = strip_bin_to_mm(v_bins, center_bin)
    s_w_mm = strip_bin_to_mm(w_bins, center_bin)

    # 4) For each t sample, solve for x,y using the 3 s_mm values
    xs = np.empty_like(s_u_mm); ys = np.empty_like(s_u_mm)
    for i in range(len(t_vals)):
        sx = [s_u_mm[i], s_v_mm[i], s_w_mm[i]]
        x_mm, y_mm = solve_xy_from_s(sx, phis)
        xs[i] = x_mm
        ys[i] = y_mm

    # 5) z from t
    zs = (t_vals - T_OFFSET_CONST - TRIGGER_DELAY) * MM_PER_TBIN

    pts = np.vstack([xs, ys, zs]).T
    center, direction = fit_line_pca(pts)
    proj = (pts - center) @ direction
    ep0 = center + direction * proj.min()
    ep1 = center + direction * proj.max()

    if verbose:
        print("Approach A (geometry A) reconstruction:")
        print("center:", center)
        print("direction:", direction)
        print("ep0_mm:", ep0)
        print("ep1_mm:", ep1)

    return dict(center=center, direction=direction, ep0_mm=ep0, ep1_mm=ep1, pts=pts, fits=((a_u,b_u),(a_v,b_v),(a_w,b_w)))

if __name__ == "__main__":
    import testing
    raw = testing.getTestData("middle")
    res = reconstruct_from_histograms(raw[0], phis=PHIS, n_samples=600)
    print("ep0_mm:", res["ep0_mm"])
    print("ep1_mm:", res["ep1_mm"])
    try:
        gt0, gt1 = raw[3], raw[4]
        print("GT:", gt0, gt1)
        print("errs:", np.linalg.norm(res["ep0_mm"] - gt0), np.linalg.norm(res["ep1_mm"] - gt1))
    except Exception:
        pass
