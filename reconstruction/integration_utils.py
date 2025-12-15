from testing import getXYZtoUVWT

def xyz_points_to_global_data(pts_xyz):
    """
    Convert XYZ points to global reconstruction input.
    Each XYZ point generates 3 residuals: U, V, W.
    """
    uvwt = getXYZtoUVWT(pts_xyz)

    data = []

    for (u, v, w, t), (x, y, z) in zip(uvwt, pts_xyz):
        data.append({
            "plane": "U",
            "u": u,
            "z": z,
            "w": 1.0,
        })
        data.append({
            "plane": "V",
            "u": v,
            "z": z,
            "w": 1.0,
        })
        data.append({
            "plane": "W",
            "u": w,
            "z": z,
            "w": 1.0,
        })

    return data
