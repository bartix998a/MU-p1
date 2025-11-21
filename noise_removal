import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pywt

# ------------------------------
# 1. Threshold-based filtering

def denoise_threshold(image, threshold):
    """Simple threshold: values below threshold → zero."""
    return np.where(image > threshold, image, 0)


# ------------------------------
# 2. Gaussian smoothing


def denoise_gaussian(image, sigma=1.0):
    """Low-pass smoothing that removes high-frequency noise."""
    return gaussian_filter(image, sigma=sigma)


# ------------------------------
# 3. Median filter

def denoise_median(image, size=3):
    """Replaces each pixel by the median of neighbors."""
    return median_filter(image, size=size)


# ------------------------------
# 4. Morphological filtering

def denoise_morphological(image, structure_size=3):
    """Opening removes small bright noise; closing fills gaps."""
    structure = np.ones((structure_size, structure_size))
    opened = binary_opening(image > 0, structure=structure)
    closed = binary_closing(opened, structure=structure)
    return closed.astype(image.dtype) * image.max()


# ------------------------------
# 5. FFT low-pass filtering

def denoise_fft_lowpass(image, cutoff=0.1):
    """Remove high-frequency (fast) noise in the Fourier domain."""
    F = fftshift(fft2(image))
    H = np.zeros_like(F)
    
    rows, cols = image.shape
    center_r, center_c = rows//2, cols//2
    radius = cutoff * min(rows, cols)

    for r in range(rows):
        for c in range(cols):
            if np.sqrt((r - center_r)**2 + (c - center_c)**2) < radius:
                H[r, c] = 1

    F_filtered = F * H
    return np.abs(ifft2(ifftshift(F_filtered)))


# ------------------------------
# 6. Wavelet denoising

def denoise_wavelet(image, wavelet="db2", level=2, threshold=0.1):
    """Threshold small wavelet coefficients → noise removed."""
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_filtered = []

    for c in coeffs:
        if isinstance(c, tuple):
            c_filtered = tuple(pywt.threshold(x, threshold*np.max(x)) for x in c)
            coeffs_filtered.append(c_filtered)
        else:
            coeffs_filtered.append(c)

    return pywt.waverec2(coeffs_filtered, wavelet)


# ------------------------------
# Combine all methods and return results

def compare_all_methods(image):
    """Applies all denoising methods and returns a dictionary of results."""
    return {
        "threshold": denoise_threshold(image, threshold=np.mean(image)),
        "gaussian": denoise_gaussian(image, sigma=1.0),
        "median": denoise_median(image, size=3),
        "morphological": denoise_morphological(image, structure_size=3),
        "fft_lowpass": denoise_fft_lowpass(image, cutoff=0.1),
        "wavelet": denoise_wavelet(image, level=2, threshold=0.1)
    }
