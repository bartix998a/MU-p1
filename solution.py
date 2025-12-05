import numpy as np
from typing import Literal
from noise_removal import denoise
from reconstruct_geometryA_uvwt import reconstruct_aligned
from dens import estimateMiddle

def solution(images :list[np.ndarray], denoising :Literal["threshold", "gaussian", "median", "morphological", "wavelet", "fft"]):
    images_denoised = np.array([denoise(image, denoising) for image in images])
    points = reconstruct_aligned(images_denoised, verbose = False)
    start, end = points["ep0_gt"], points["ep1_gt"]
    middle = estimateMiddle(images_denoised, start, end)
    return start, end, middle