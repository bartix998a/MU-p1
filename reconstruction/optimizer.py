import numpy as np
from reconstruction.residuals import residual_and_jacobian
from reconstruction.geometry import normalize

def optimize_global_line_LM(data, initial_params,
                            max_iter=30,
                            lambda0=1e-3,
                            eps_param=1e-6,
                            eps_loss=1e-6):

    p = initial_params.copy()
    lambda_ = lambda0

    r, J, w = residual_and_jacobian(p, data)
    W = np.diag(w)
    loss = r.T @ W @ r

    for _ in range(max_iter):
        H = J.T @ W @ J
        g = J.T @ W @ r

        try:
            delta = np.linalg.solve(
                H + lambda_ * np.eye(len(p)), g
            )
        except np.linalg.LinAlgError:
            lambda_ *= 10
            continue

        p_trial = p + delta

        # enforce unit direction
        p_trial[:3] = normalize(p_trial[:3])

        r_new, J_new, _ = residual_and_jacobian(p_trial, data)
        loss_new = r_new.T @ W @ r_new

        if loss_new < loss:
            p = p_trial
            r, J = r_new, J_new
            loss = loss_new
            lambda_ *= 0.3

            if np.linalg.norm(delta) < eps_param:
                break
            if abs(loss_new - loss) < eps_loss:
                break
        else:
            lambda_ *= 10

    return p, r, J, W