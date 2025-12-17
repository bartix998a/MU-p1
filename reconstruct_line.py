"""
Public entry point for line reconstruction.

reconstruct_line(images) returns a dict with:
- status: OK | FALLBACK | FAILED
- ep0_mm, ep1_mm
- direction
- uncertainty (if available)
- flags
"""

import numpy as np
from reconstruction.geometry import normalize
from reconstruction.global_line import reconstruct_with_uncertainty
from reconstruction.integration_utils import xyz_points_to_global_data
from reconstruct_geometryA_uvwt_legacy import reconstruct_from_histograms_notebook

def reconstruct_line(images):
    histU, histV, histW = images
    raw0 = ((histU, histV, histW), None, None)
    legacy = reconstruct_from_histograms_notebook((raw0, None, None, None, None))
    pts_xyz = legacy["pts"]

    data = xyz_points_to_global_data(pts_xyz)

    init = np.array([0, 0, 1, 0, 0], float)
    res = reconstruct_with_uncertainty(data, init)

    if res["status"] == "OK" and not res["flags"]:
        d = normalize(res["params"][:3])
        center = 0.5 * (p0 + p1)
        half_len = 0.5 * np.linalg.norm(p1 - p0)
        p0 = np.asarray(legacy["ep0_mm"], float)
        p1 = np.asarray(legacy["ep1_mm"], float)
        t0 = np.dot(p0 - center, d)
        t1 = np.dot(p1 - center, d)

        tmin, tmax = min(t0, t1), max(t0, t1)

        ep0 = center + tmin * d
        ep1 = center + tmax * d



        return {
            "status": "OK",
            "ep0_mm": ep0,
            "ep1_mm": ep1,
            "direction": d,
            "uncertainty": res["endpoint_uncertainty"],
            "flags": res["flags"],
        }

    return {
        "status": "FALLBACK",
        "ep0_mm": legacy["ep0_mm"],
        "ep1_mm": legacy["ep1_mm"],
        "direction": legacy["direction"],
        "uncertainty": None,
        "flags": ["USED_LEGACY"],
    }

if __name__ == "__main__":
    print("reconstruct_line module loaded successfully")
