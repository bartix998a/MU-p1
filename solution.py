import numpy as np
from typing import Literal
from noise_removal import denoise
from reconstruct_geometryA_uvwt import reconstruct_aligned
from dens import estimateMiddle
from testing import getTestData

def solution(images :list[np.ndarray], denoising :Literal["threshold", "gaussian", "median", "morphological", "wavelet", "fft"]):
    images_denoised = np.array([denoise(image, denoising) for image in images])
    points = reconstruct_aligned(images_denoised, verbose = False)
    start, end = points["ep0_gt"], points["ep1_gt"]
    middle = estimateMiddle(images_denoised, start, end)
    return start, end, middle

def estimateAccuracy(n_calls = 1000):
    responses = []
    actual_vals = []
    
    for i in range(n_calls):
        histograms, start, end, middle = getTestData('all') # type:ignore
        responses += [solution(histograms, denoising='gaussian')]
        actual_vals += [np.array([start, middle, end])]
        
    responses = np.array(responses)
    actual_vals = np.array(actual_vals)
    
    return np.average(np.linalg.norm(responses - actual_vals, axis = 1))

estimateAccuracy()