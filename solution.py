import numpy as np
from typing import Literal
from noise_removal import denoise
from reconstruct_geometryA_uvwt_legacy import reconstruct_from_histograms_notebook
from reconstruct_line import reconstruct_line
from dens import estimateMiddle
from testing import getTestData

def solution(images :list[np.ndarray],
             denoising :Literal["threshold", "gaussian", "median", "morphological", "wavelet", "fft"]):

    images_denoised = np.array([denoise(image, denoising) for image in images])

    # NEW reconstruction (with fallback inside)
    line = reconstruct_line(images_denoised)

    start, end = line["ep0_mm"], line["ep1_mm"]
    middle = estimateMiddle(images_denoised, start, end)

    return start, end, middle


def estimateAccuracy(n_calls = 100,
                     where :Literal['noise', 'fit', 'edges', 'middle', 'all'] = 'all'):

    responses = []
    actual_vals = []
    results = []

    for i in range(n_calls):
        if where == 'noise':
            histograms, clear_histograms = getTestData('noise')
            histograms = np.array([[denoise(image, 'gaussian') for image in histograms]])
            results += [np.linalg.matrix_norm(hist[0] - hist[1]) for hist in zip(histograms, clear_histograms)]

        elif where == 'fit':
            hist, start_gt, end_gt = getTestData('fit')
            points = reconstruct_from_histograms_notebook(((hist, start_gt, end_gt), None, None, None, None))
            start, end = points["ep0_mm"], points["ep1_mm"]
            results += [
                min(np.linalg.norm(start - gt) for gt in [start_gt, end_gt]),
                min(np.linalg.norm(end - gt) for gt in [start_gt, end_gt])
            ]
        elif where == 'global_fit':
            hist, start_gt, end_gt = getTestData('fit')
            start, end, _ = solution(hist, denoising='gaussian')

            results += [
                min(np.linalg.norm(start - gt) for gt in [start_gt, end_gt]),
                min(np.linalg.norm(end - gt) for gt in [start_gt, end_gt])
            ]
        elif where == 'middle':
            data, vertex = getTestData('middle')
            results += [np.linalg.norm(vertex, estimateMiddle(*data))]

        else:
            histograms, start, end, middle = getTestData('all')
            responses += [solution(histograms, denoising='gaussian')]
            actual_vals += [np.array([start, middle, end])]
            results += [np.linalg.norm(responses[-1] - actual_vals[-1], axis = 1)]

    responses = np.array(responses)
    actual_vals = np.array(actual_vals)

    return np.average(np.array(results))


legacy_err = estimateAccuracy(where='fit')
global_err = estimateAccuracy(where='global_fit')

print("Legacy endpoint error:", legacy_err)
print("Global  endpoint error:", global_err)
print("Improvement factor:", legacy_err / global_err if global_err > 0 else np.inf)