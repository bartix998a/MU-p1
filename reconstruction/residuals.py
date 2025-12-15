import numpy as np
from reconstruction.geometry import normalize, orthonormal_basis, project_to_strip

SIGMA_STRIP = 1.0
SIGMA_Z = 1.0

def residual_and_jacobian(params, data):
    """
    params = (dx, dy, dz, a, b)
    """
    # ---- unpack parameters ----
    dx, dy, dz, a, b = params

    # direction: normalize every evaluation
    d = normalize(np.array([dx, dy, dz]))

    # build orthonormal basis for offset
    e1, e2 = orthonormal_basis(d)
    x0 = a * e1 + b * e2

    residuals = []
    J = []
    weights = []

    for row in data:
        plane = row["plane"]
        z     = row["z"]
        u_obs = row["u"]

        # ---- degeneracy guard (same logic as before) ----
        if abs(d[2]) < 1e-8:
            continue

        # ---- line–plane intersection at given z ----
        s = (z - x0[2]) / d[2]
        pt = x0 + s * d

        # ---- predicted strip coordinate ----
        u_pred = project_to_strip(pt[0], pt[1], plane)

        # ---- residual ----
        r = u_obs - u_pred
        residuals.append(r)
        effective_sigma = np.sqrt(
            SIGMA_STRIP ** 2 + (SIGMA_Z / abs(d[2])) ** 2
        )
        weights.append(1.0 / effective_sigma ** 2)

        # ---- numerical Jacobian ----
        eps = 1e-6
        J_row = []

        for k in range(len(params)):
            dp = np.zeros_like(params)
            dp[k] = eps

            p2 = params + dp

            # IMPORTANT: renormalize direction for perturbed params
            d2 = normalize(p2[:3])

            e1_2, e2_2 = orthonormal_basis(d2)
            x0_2 = p2[3] * e1_2 + p2[4] * e2_2

            if abs(d2[2]) < 1e-8:
                J_row.append(0.0)
                continue

            s2 = (z - x0_2[2]) / d2[2]
            pt2 = x0_2 + s2 * d2

            u2 = project_to_strip(pt2[0], pt2[1], plane)

            J_row.append((u2 - u_pred) / eps)

        J.append(J_row)

    return (
        np.array(residuals),
        np.array(J),
        np.array(weights)
    )
