import numpy as np
from reconstruction.geometry import normalize, orthonormal_basis

def endpoint_from_phys_params(p_phys, d, s):
    """
    p_phys = (u1, u2, a, b)
    where u1,u2 are direction perturbations along tangent basis
    """
    u1, u2, a, b = p_phys

    # direction perturbation
    t1, t2 = orthonormal_basis(d)
    d_pert = normalize(d + u1 * t1 + u2 * t2)

    x0 = a * t1 + b * t2
    return x0 + s * d_pert

def project_hessian_to_physical_space(H, params):
    """
    Project 5x5 Hessian in (dx,dy,dz,a,b) space
    onto the 4D physical parameter space:
      (2 direction DOF + a + b)

    Returns:
        H_phys : (4,4) ndarray
        P      : projection matrix (4x5)
    """
    d = normalize(params[:3])

    # Tangent basis on S^2
    t1, t2 = orthonormal_basis(d)

    # Projection matrix
    P = np.zeros((4, 5))

    # Direction DOFs
    P[0, :3] = t1
    P[1, :3] = t2

    # Offset DOFs
    P[2, 3] = 1.0
    P[3, 4] = 1.0

    H_phys = P @ H @ P.T
    return H_phys, P

def endpoint_covariance(d, cov_phys, s, eps=1e-6):
    """
    Propagate parameter covariance to endpoint at position s.

    d         : unit direction (3,)
    cov_phys  : 4x4 covariance (tangent1, tangent2, a, b)
    s         : scalar line parameter
    """
    # reference endpoint
    p0 = np.zeros(4)
    x0 = endpoint_from_phys_params(p0, d, s)

    J = np.zeros((3, 4))

    for i in range(4):
        dp = np.zeros(4)
        dp[i] = eps

        x1 = endpoint_from_phys_params(dp, d, s)
        J[:, i] = (x1 - x0) / eps

    # covariance propagation
    return J @ cov_phys @ J.T

def endpoint_sigma(cov_ep):
    """
    Scalar endpoint uncertainty (RMS)
    """
    return np.sqrt(np.trace(cov_ep))
