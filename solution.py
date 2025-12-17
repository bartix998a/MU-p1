import numpy as np
from typing import Literal

from noise_removal import denoise
from reconstruct_line import reconstruct_line
from dens import estimateMiddle
from testing import getTestData


def solution(
    images: list[np.ndarray],
    denoising: Literal[
        "threshold", "gaussian", "median",
        "morphological", "wavelet", "fft"
    ] = "gaussian",
):
    # --- denoise ---
    images_denoised = np.array([
        denoise(image, denoising) for image in images
    ])

    # --- reconstruct (global + fallback happens inside) ---
    line = reconstruct_line(images_denoised)

    if line is None or "ep0_mm" not in line:
        raise RuntimeError("Reconstruction failed completely")

    start = np.asarray(line["ep0_mm"], float)
    end   = np.asarray(line["ep1_mm"], float)

    # --- endpoint sanity (CRITICAL) ---
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise RuntimeError("NaN endpoints")

    # consistent ordering to avoid sign chaos
    if start[2] > end[2]:
        start, end = end, start

    middle = estimateMiddle(images_denoised, start, end)
    return start, end, middle


def estimateAccuracy(
    n_calls: int = 100,
    where: Literal["all"] = "all",
):
    results = []

    for _ in range(n_calls):
        hist, start_gt, end_gt = getTestData("fit")

        try:
            start, end, _ = solution(hist, denoising="gaussian")
        except Exception:
            # fallback: use legacy directly for evaluation only
            from reconstruct_geometryA_uvwt_legacy import reconstruct_from_histograms_notebook
            pts = reconstruct_from_histograms_notebook(((hist, None, None), None, None, None, None))
            start, end = pts["ep0_mm"], pts["ep1_mm"]

        results.append(
            min(np.linalg.norm(start - gt) for gt in [start_gt, end_gt])
        )
        results.append(
            min(np.linalg.norm(end - gt) for gt in [start_gt, end_gt])
        )

    if not results:
        return np.inf

    return float(np.mean(results))


if __name__ == "__main__":
    err = estimateAccuracy(where="all")
    print("========== SUMMARY ==========")
    print("Hybrid reconstruction (global + legacy fallback)")
    print("Mean endpoint error:", err)
