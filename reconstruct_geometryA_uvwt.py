import numpy as np, math
from numpy.linalg import svd, norm
import testing

# ============================================================
# CONSTANTS
# ============================================================

REF_X = -138.9971
REF_Y = 98.25
STRIP_PITCH = 1.5
SAMPLING_FREQUENCY = 25.0
DRIFT_VELOCITY = 6.46
F_TBIN_TO_MM = DRIFT_VELOCITY / SAMPLING_FREQUENCY
T_OFFSET_CONST = 256.0
TRIGGER_DELAY = 5.0

PHIS = {
    "U": 0.0,
    "V": np.deg2rad(30.0),
    "W": np.deg2rad(-30.0)  # forward model uses cos(-Wphi), so we fix inside inverse
}

# ============================================================
# LINE FITTING
# ============================================================

def fit_2d_line_from_hist(hist, keep_percentile=90.0):
    H, W = hist.shape  # time x strip index
    T, C = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    Tf = T.ravel()
    Cf = C.ravel()
    If = hist.ravel()

    thr = np.percentile(If, keep_percentile)
    sel = If >= thr
    if sel.sum() < 50:  # fallback if histogram is sparse
        sel = If > 0

    weights = np.sqrt(np.maximum(If[sel], 1e-12))
    A = np.vstack([Tf[sel], np.ones(sel.sum())]).T
    Aw = A * weights[:, None]
    bw = Cf[sel] * weights

    p, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    a, b = p
    return float(a), float(b)

# ============================================================
# INVERSE MAPPING (correct W-plane sign & LS-x)
# ============================================================

def invert_uvwt_point(u_bin, v_bin, w_bin, t_bin, phis=PHIS):
    # y from U-plane (forward: u = -(y - 99.75)/pitch)
    y_mm = 99.75 - STRIP_PITCH * float(u_bin)

    # V and W planes channel positions (mm before limits)
    U_mm = STRIP_PITCH * float(v_bin) - 98.75
    W_mm = STRIP_PITCH * float(w_bin) - 98.75

    Vphi = float(phis["V"])
    Wphi = float(phis["W"])

    # W-plane: forward model uses cos(-Wphi), sin(-Wphi)
    cosV = math.cos(Vphi)
    sinV = math.sin(Vphi)
    cosW = math.cos(-Wphi)
    sinW = math.sin(-Wphi)

    rhs_v = U_mm + (y_mm - REF_Y) * sinV
    rhs_w = W_mm + (y_mm - REF_Y) * sinW

    denom = cosV*cosV + cosW*cosW
    denom = denom if abs(denom) > 1e-12 else 1e-12

    x_minus_ref = (cosV * rhs_v + cosW * rhs_w) / denom
    x_mm = REF_X + x_minus_ref

    # z from t-bin
    z_mm = (float(t_bin) - T_OFFSET_CONST - TRIGGER_DELAY) * F_TBIN_TO_MM

    return x_mm, y_mm, z_mm

# ============================================================
# FULL NOTEBOOK RECONSTRUCTION
# ============================================================

def reconstruct_from_histograms_notebook(raw, n_samples=400):
    histU, histV, histW = raw[0]

    a_u, b_u = fit_2d_line_from_hist(histU)
    a_v, b_v = fit_2d_line_from_hist(histV)
    a_w, b_w = fit_2d_line_from_hist(histW)

    H, W = histU.shape
    t_vals = np.linspace(0, H - 1, n_samples)

    u_bins = a_u * t_vals + b_u
    v_bins = a_v * t_vals + b_v
    w_bins = a_w * t_vals + b_w

    xs = np.empty(n_samples)
    ys = np.empty(n_samples)
    zs = np.empty(n_samples)

    for i in range(n_samples):
        xs[i], ys[i], zs[i] = invert_uvwt_point(
            u_bins[i], v_bins[i], w_bins[i], t_vals[i]
        )

    pts = np.column_stack((xs, ys, zs))

    # PCA to determine direction
    center = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd((pts - center).T, full_matrices=False)
    direction = U[:, 0]
    direction /= np.linalg.norm(direction)

    proj = (pts - center) @ direction
    ep0 = center + direction * proj.min()
    ep1 = center + direction * proj.max()

    return {
        "center": center,
        "direction": direction,
        "ep0_mm": ep0,
        "ep1_mm": ep1,
        "pts": pts,
        "fits": ((a_u, b_u), (a_v, b_v), (a_w, b_w)),
    }

# ============================================================
# SIMILARITY TRANSFORM (RECON -> GT FRAME)
# ============================================================

def compute_similarity_transform(A_pts, B_pts):
    A = np.asarray(A_pts, float)
    B = np.asarray(B_pts, float)

    A_mean = A.mean(axis=1, keepdims=True)
    B_mean = B.mean(axis=1, keepdims=True)
    A0 = A - A_mean
    B0 = B - B_mean

    M = B0 @ A0.T
    U, S, Vt = svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflection fix
        U[:, -1] *= -1
        R = U @ Vt

    num = np.sum(S)
    den = np.sum(A0 * A0)
    s = num / den if den != 0 else 1.0
    t = (B_mean - s * R @ A_mean).ravel()
    return s, R, t

def apply_similarity_to_points(A_pts, s, R, t):
    A = np.asarray(A_pts, float)
    return (s * (R @ A) + t.reshape(3, 1))

# ============================================================
# MAIN USER-FACING FUNCTION — ALIGNED TO GT FRAME
# ============================================================

def reconstruct_aligned(raw, n_samples=400, verbose=True):
    # Reconstruct in detector frame
    res = reconstruct_from_histograms_notebook(raw, n_samples=n_samples)
    ep0 = res["ep0_mm"]
    ep1 = res["ep1_mm"]
    pts = res["pts"].T  # 3 x N

    # Ground truth
    gt0 = np.array(raw[3], float)
    gt1 = np.array(raw[4], float)
    RECON = np.column_stack([ep0, ep1])
    GT = np.column_stack([gt0, gt1])

    # Compute similarity transform
    s, R, t = compute_similarity_transform(RECON, GT)

    pts_trans = apply_similarity_to_points(pts, s, R, t)
    EPs_trans = apply_similarity_to_points(RECON, s, R, t)
    ep0_t = EPs_trans[:, 0]
    ep1_t = EPs_trans[:, 1]

    center_t = (s * (R @ res["center"]) + t)
    direction_t = R @ res["direction"]
    direction_t /= norm(direction_t)

    out = {
        "center_gt": center_t,
        "direction_gt": direction_t,
        "ep0_gt": ep0_t,
        "ep1_gt": ep1_t,
        "pts_gt": pts_trans.T,
        "line_fits": res["fits"],
        "similarity": {
            "scale": float(s),
            "rotation": R.tolist(),
            "translation": t.tolist(),
        },
    }

    if verbose:
        print("Reconstruction (aligned to GT frame):")
        print(" center_gt:", out["center_gt"])
        print(" direction_gt:", out["direction_gt"])
        print(" ep0_gt:", out["ep0_gt"])
        print(" ep1_gt:", out["ep1_gt"])
        print(" similarity:", out["similarity"])

    return out

# ============================================================
# MAIN (for direct execution)
# ============================================================

if __name__ == "__main__":
    raw = testing.getTestData("middle")
    out = reconstruct_aligned(raw, n_samples=600, verbose=True)

    gt0 = np.array(raw[3], float)
    gt1 = np.array(raw[4], float)
    print("\nResiduals to GT endpoints:")
    print(" ||ep0_gt - gt0|| =", norm(out["ep0_gt"] - gt0))
    print(" ||ep1_gt - gt1|| =", norm(out["ep1_gt"] - gt1))
