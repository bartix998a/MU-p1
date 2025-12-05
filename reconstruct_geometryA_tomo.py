# reconstruct_geometryA_tomo.py
"""
Approach C — matrix-free volumetric accumulation (diagnostic tomography) for geometry A.
This is intentionally simple: with only 3 projections tomography is under-determined,
but accumulation + skeleton + PCA can give a coarse estimate.
"""

import numpy as np
from numpy.linalg import svd
try:
    from skimage.morphology import skeletonize_3d
    SKIMAGE = True
except Exception:
    SKIMAGE = False

# constants
STRIP_PITCH = 1.5
SAMPLING_FREQUENCY = 25.0
DRIFT_VELOCITY = 6.46
MM_PER_TBIN = DRIFT_VELOCITY / SAMPLING_FREQUENCY
T_OFFSET_CONST = 256.0
TRIGGER_DELAY = 5.0

PHIS = {
    "U": 0.0,
    "V": np.deg2rad(+30.0),
    "W": np.deg2rad(-30.0)
}

def plane_axes_for_phi(phi):
    # For geometry A, projection axis is unit vector in XY plane at angle phi
    ux = np.cos(phi); uy = np.sin(phi)
    # axis_u (time direction) points along Z
    axis_u = np.array([0.0, 0.0, 1.0])
    # axis_v (strip axis) oriented in XY plane at phi
    axis_v = np.array([ux, uy, 0.0])
    origin = np.zeros(3, dtype=float)
    return origin, axis_u, axis_v

def reconstruct_volume_accumulate(hist_list, phis=PHIS, voxel_res=96,
                                  xlim=(-100,100), ylim=(-100,100), zlim=(-100,100)):
    # build voxel grid
    xs = np.linspace(xlim[0], xlim[1], voxel_res)
    ys = np.linspace(ylim[0], ylim[1], voxel_res)
    zs = np.linspace(zlim[0], zlim[1], voxel_res)
    vol = np.zeros((voxel_res, voxel_res, voxel_res), dtype=float)

    # iterate projections
    for k, key in enumerate(['U','V','W']):
        phi = float(phis[key])
        origin, axis_u, axis_v = plane_axes_for_phi(phi)
        h = hist_list[k]
        H, W = h.shape
        thr = np.percentile(h.ravel(), 98.0)
        idxs = np.where(h >= thr)
        for ti, ci in zip(idxs[0], idxs[1]):
            intensity = float(h[ti, ci])
            # map to mm on plane
            t_mm = ti * (DRIFT_VELOCITY / SAMPLING_FREQUENCY)
            s_mm = (ci - (W/2.0)) * STRIP_PITCH
            pt = origin + axis_u * t_mm + axis_v * s_mm
            # deposit into nearest voxel (simple)
            ix = np.searchsorted(xs, pt[0]); iy = np.searchsorted(ys, pt[1]); iz = np.searchsorted(zs, pt[2])
            if 0 <= ix < voxel_res and 0 <= iy < voxel_res and 0 <= iz < voxel_res:
                vol[ix, iy, iz] += intensity
    return vol, xs, ys, zs

def skeleton_and_fit(vol, xs, ys, zs):
    norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-12)
    thr = np.percentile(norm, 99.0)
    mask = norm >= thr
    if mask.sum() == 0:
        thr = np.percentile(norm, 97.0)
        mask = norm >= thr
        if mask.sum() == 0:
            return None
    if SKIMAGE:
        sk = skeletonize_3d(mask.astype(np.uint8))
        coords = np.argwhere(sk > 0)
    else:
        coords = np.argwhere(mask)
    if coords.shape[0] < 5:
        return None
    pts_mm = np.vstack([xs[coords[:,0]], ys[coords[:,1]], zs[coords[:,2]]]).T
    center = pts_mm.mean(axis=0)
    U,S,Vt = svd((pts_mm - center).T, full_matrices=False)
    dirv = U[:,0]; dirv /= (np.linalg.norm(dirv)+1e-12)
    proj = (pts_mm - center) @ dirv
    ep0 = center + dirv * proj.min()
    ep1 = center + dirv * proj.max()
    return dict(center=center, dir=dirv, ep0=ep0, ep1=ep1, pts=pts_mm)

if __name__ == "__main__":
    import testing
    raw = testing.getTestData('middle')
    hist_list = raw[0]
    vol, xs, ys, zs = reconstruct_volume_accumulate(hist_list, PHIS, voxel_res=96)
    res = skeleton_and_fit(vol, xs, ys, zs)
    if res is None:
        print("No skeleton/line extracted")
    else:
        print("ep0:", res['ep0'], "ep1:", res['ep1'])
        try:
            gt0, gt1 = raw[3], raw[4]
            print("GT:", gt0, gt1)
            print("errs:", np.linalg.norm(res['ep0'] - gt0), np.linalg.norm(res['ep1'] - gt1))
        except Exception:
            pass
