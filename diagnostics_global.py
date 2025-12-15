import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reconstruction.global_line import reconstruct_with_uncertainty
from reconstruction.integration_utils import xyz_points_to_global_data
from reconstruction.geometry import normalize

# =========================
# 1. Synthetic data generation
# =========================

def sample_directions_random(n):
    v = np.random.normal(size=(n, 3))
    return np.array([normalize(x) for x in v])

def sample_directions_uniform(n):
    phi = np.random.uniform(0, 2*np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)
    return np.stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ], axis=1)

def sample_lengths(n, Lmin=10, Lmax=200):
    return np.exp(np.random.uniform(np.log(Lmin), np.log(Lmax), n))

def sample_anchors(n, box=50):
    return np.random.uniform(-box, box, size=(n, 3))

def generate_xyz_hits(x0, d, L, n_hits=80, noise_xyz=0.0):
    s = np.linspace(-L/2, L/2, n_hits)
    pts = x0[None, :] + s[:, None] * d[None, :]
    if noise_xyz > 0:
        pts += np.random.normal(scale=noise_xyz, size=pts.shape)
    return pts

# =========================
# 2. Metrics
# =========================

def direction_error(d_true, d_rec):
    return np.arccos(
        np.clip(abs(np.dot(normalize(d_true), normalize(d_rec))), -1, 1)
    )

# =========================
# 3. Ensemble evaluation
# =========================

def run_ensemble(
    n_lines=500,
    direction_mode="random",   # "random" or "uniform"
    noise_xyz=0.0,
    seed=0
):
    np.random.seed(seed)

    dirs = (
        sample_directions_random(n_lines)
        if direction_mode == "random"
        else sample_directions_uniform(n_lines)
    )

    lengths = sample_lengths(n_lines)
    anchors = sample_anchors(n_lines)

    results = []

    for i in range(n_lines):
        pts_xyz = generate_xyz_hits(
            anchors[i], dirs[i], lengths[i], noise_xyz=noise_xyz
        )

        data = xyz_points_to_global_data(pts_xyz)
        init = np.array([0, 0, 1, 0, 0], float)

        res = reconstruct_with_uncertainty(data, init)

        entry = {
            "true_dir": dirs[i],
            "true_L": lengths[i],
            "dz": abs(dirs[i][2]),

            "status": res["status"],
            "flags": res.get("flags", []),
            "condition": res.get("condition"),
            "endpoint_uncertainty": res.get("endpoint_uncertainty"),
            "rec_dir": (
                normalize(res["params"][:3])
                if res["status"] == "OK"
                else None
            ),
        }

        results.append(entry)

    return results

# =========================
# 4. Plots
# =========================

def plot_direction_error_vs_length(results):
    L, err = [], []
    for r in results:
        if r["status"] == "OK":
            L.append(r["true_L"])
            err.append(direction_error(r["true_dir"], r["rec_dir"]))

    plt.figure()
    plt.scatter(L, err, s=10, alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Track length")
    plt.ylabel("Direction error [rad]")
    plt.title("Global direction error vs track length")
    plt.grid(True)
    plt.savefig("./plots/direction_error_vs_length.png", dpi=200)
    plt.close()

def plot_direction_error_vs_length_gated(results, dz_min=0.5):
    L, err = [], []
    for r in results:
        if r["status"] == "OK" and r["dz"] > dz_min:
            L.append(r["true_L"])
            err.append(direction_error(r["true_dir"], r["rec_dir"]))

    plt.figure()
    plt.scatter(L, err, s=10, alpha=0.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Track length")
    plt.ylabel("Direction error [rad]")
    plt.title(f"Global direction error (|dz| > {dz_min})")
    plt.grid(True)
    plt.savefig("./plots/direction_error_vs_length_gated.png", dpi=200)
    plt.close()

def plot_condition_vs_dz(results):
    dz, cond = [], []
    for r in results:
        if r["status"] == "OK":
            dz.append(r["dz"])
            cond.append(r["condition"])

    plt.figure()
    plt.scatter(dz, cond, s=10, alpha=0.6)
    plt.yscale("log")
    plt.xlabel("|dz|")
    plt.ylabel("Condition number")
    plt.title("Condition number vs |dz|")
    plt.grid(True)
    plt.savefig("./plots/condition_vs_dz.png", dpi=200)
    plt.close()

def plot_flag_distribution(results):
    ok, soft, hard = 0, 0, 0
    for r in results:
        if r["status"] != "OK":
            hard += 1
        elif r["flags"]:
            soft += 1
        else:
            ok += 1

    plt.figure()
    plt.bar(["OK", "SOFT", "HARD"], [ok, soft, hard])
    plt.title("Global reconstruction outcome")
    plt.ylabel("Count")
    plt.savefig("./plots/flag_distribution.png", dpi=200)
    plt.close()

# =========================
# 5. Summary
# =========================

def summarize(results):
    err = [
        direction_error(r["true_dir"], r["rec_dir"])
        for r in results if r["status"] == "OK"
    ]

    print("========== SUMMARY ==========")
    print("Total tracks:", len(results))
    print("OK:", sum(r["status"] == "OK" for r in results))
    print("Soft-flagged:", sum(bool(r["flags"]) for r in results))
    print("Mean direction error:", np.mean(err))
    print("Median direction error:", np.median(err))
    print("95% percentile:", np.percentile(err, 95))
    print("Fraction |dz| < 0.2:", np.mean([r["dz"] < 0.2 for r in results]))

# =========================
# 6. Main
# =========================

def main():
    results = run_ensemble(
        n_lines=500,
        direction_mode="random",
        noise_xyz=0.0
    )

    summarize(results)

    plot_direction_error_vs_length(results)
    plot_direction_error_vs_length_gated(results)
    plot_condition_vs_dz(results)
    plot_flag_distribution(results)

if __name__ == "__main__":
    main()
