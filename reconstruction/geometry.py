import numpy as np

STRIP_PITCH = 1.5
REF_X = -138.9971
REF_Y = 98.25
PHIS = {"U": 0.0, "V": np.deg2rad(30.0), "W": np.deg2rad(-30.0)}

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def direction_from_cartesian(d):
    """
    Convert unconstrained Cartesian direction parameters
    into a unit direction vector.
    """
    return normalize(d)

def orthonormal_basis(d):
    """
    Return two unit vectors orthogonal to direction d.
    Assumes d is unit length.
    """
    if abs(d[2]) < 0.9:
        e1 = np.cross(d, np.array([0.0, 0.0, 1.0]))
    else:
        e1 = np.cross(d, np.array([0.0, 1.0, 0.0]))
    e1 = normalize(e1)
    e2 = np.cross(d, e1)
    return e1, e2

def project_to_strip(x, y, plane):
    if plane == "U":
        return -(y - 99.75) / STRIP_PITCH

    phi = PHIS[plane]
    if plane == "V":
        return ((x - REF_X) * np.cos(phi) - (y - REF_Y) * np.sin(phi) + 98.75) / STRIP_PITCH
    if plane == "W":
        return ((x - REF_X) * np.cos(-phi) - (y - REF_Y) * np.sin(-phi) + 98.75) / STRIP_PITCH

    raise ValueError("Unknown plane")