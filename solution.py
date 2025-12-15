import numpy as np
from typing import Literal
from noise_removal import denoise
from reconstruct_geometryA_uvwt import reconstruct_from_histograms_notebook
from dens import estimateMiddle
from testing import getTestData

def solution(images :list[np.ndarray], denoising :Literal["threshold", "gaussian", "median", "morphological", "wavelet", "fft"]):
    images_denoised = np.array([denoise(image, denoising) for image in images])
    points = reconstruct_from_histograms_notebook((images_denoised, None, None, None, None))
    start, end = points["ep0_mm"], points["ep1_mm"]
    middle = estimateMiddle(images_denoised, start, end)
    return start, end, middle

def estimateAccuracy(n_calls = 2000, where :Literal['noise', 'fit', 'edges', 'middle', 'all'] = 'all'):
    responses = []
    actual_vals = []
    results = []
    
    vertex = np.random.default_rng().random(3) 
    
    for i in range(n_calls):
        if where == 'noise':
            histograms, clear_histograms = getTestData('noise') #type: ignore
            histograms = np.array([[denoise(image, 'gaussian') for image in histograms]])
            results += [np.linalg.matrix_norm(hist[0] - hist[1]) for hist in zip(histograms, clear_histograms)]
        elif where == 'fit':
            hist, start_gt, end_gt = getTestData('fit') #type: ignore
            points = reconstruct_from_histograms_notebook(((hist, start_gt, end_gt), None, None, None, None))
            start, end = points["ep0_mm"], points["ep1_mm"]
            results += [min([np.linalg.norm(start - gt)] for gt in [start_gt, end_gt]), min([np.linalg.norm(end - gt)] for gt in [start_gt, end_gt])]
        elif where == 'middle':
            data, vertex = getTestData('middle', vertex=vertex) #type: ignore
            results += [np.linalg.norm(vertex - estimateMiddle(*data))]
        else:
            histograms, start, end, middle = getTestData('all') # type:ignore
            responses += [solution(histograms, denoising='gaussian')]
            actual_vals += [np.array([start, middle, end])]
            results += [np.linalg.norm(responses[-1] - actual_vals[-1], axis = 1)]
        
    responses = np.array(responses)
    actual_vals = np.array(actual_vals)
    
    return np.average(np.array(results)), np.var(results), np.array(results).max()

print(estimateAccuracy(where='middle'))