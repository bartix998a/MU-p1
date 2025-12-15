import numpy as np
from reconstruction.optimizer import optimize_global_line_LM
from reconstruction.uncertainty import (project_hessian_to_physical_space,
                         endpoint_covariance,
                         endpoint_sigma)
from reconstruction.geometry import normalize

ENDPOINT_SIGMA_MAX = 7.5

def fit_global_line(data, initial_params):

    try:
        params, r, J, W = optimize_global_line_LM(data, initial_params)
    except Exception as e:
        return None, {"flags": ["OPTIMIZATION_FAILED"]}

    # Build full Hessian
    H = J.T @ W @ J

    # Project to physical space
    H_phys, P = project_hessian_to_physical_space(H, params)

    # Eigen-analysis in physical space
    eigs = np.linalg.eigvalsh(H_phys)

    min_eig = np.min(eigs)
    max_eig = np.max(eigs)

    condition = max_eig / max(min_eig, 1e-15)

    flags = []
    if min_eig < 1e-6:
        flags.append("HESSIAN_DEGENERATE")
    if condition > 1e8:
        flags.append("ILL_CONDITIONED")

    # Covariance in physical space
    try:
        cov_phys = np.linalg.inv(
            H_phys + 1e-12 * np.eye(4)
        )
    except np.linalg.LinAlgError:
        return None, {"flags": flags + ["INVERSION_FAILED"]}

    d = normalize(params[:3])

    # choose s-range (from data)
    z_vals = [row["z"] for row in data]
    z_span = max(z_vals) - min(z_vals)

    if abs(d[2]) < 1e-6:
        # horizontal already handled by hard flags
        s0 = s1 = 0.0
    else:
        track_len = z_span / abs(d[2])
        s0 = -0.5 * track_len
        s1 = 0.5 * track_len

    cov_ep0 = endpoint_covariance(d, cov_phys, s0)
    cov_ep1 = endpoint_covariance(d, cov_phys, s1)

    sigma_ep0 = endpoint_sigma(cov_ep0)
    sigma_ep1 = endpoint_sigma(cov_ep1)

    soft_flags = []
    if max(sigma_ep0, sigma_ep1) > ENDPOINT_SIGMA_MAX:
        soft_flags.append("LARGE_ENDPOINT_UNCERTAINTY")

    # Optional: lift covariance back to 5D parameter space
    cov_full = P.T @ cov_phys @ P

    return {
        "params": params,
        "cov": cov_full,
        "cov_phys": cov_phys,
        "eigenvalues": eigs,
        "condition": condition,
        "flags": flags,
        "endpoint_uncertainty": {
            "ep0": sigma_ep0,
            "ep1": sigma_ep1,
        },
    }, None


def reconstruct_with_uncertainty(data, initial_params):
    fit, err = fit_global_line(data, initial_params)
    if fit is None:
        return {"status": "FAILED", **err}

    return {
        "status": "OK",
        "params": fit["params"],
        "covariance": fit["cov"],
        "eigenvalues": fit["eigenvalues"],
        "condition": fit["condition"],
        "flags": fit["flags"],
        "endpoint_uncertainty": fit["endpoint_uncertainty"],
    }