import numpy as np
import cv2
import pywt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter, median_filter
from scipy import ndimage


# -------------------------------------------
# Utility metrics
# -------------------------------------------
def compute_metrics(original, denoised):
    psnr = peak_signal_noise_ratio(original, denoised, data_range=original.max() - original.min())
    ssim = structural_similarity(original, denoised)
    snr = 10 * np.log10(np.sum(original**2) / np.sum((original - denoised)**2 + 1e-9))
    return {"PSNR": psnr, "SSIM": ssim, "SNR": snr}


# -------------------------------------------
# Denoising algorithms
# -------------------------------------------

def threshold_filter(image, threshold):
    """Simple amplitude threshold"""
    out = np.copy(image)
    out[out < threshold] = 0
    return out


def gaussian_smoothing(image, sigma=1.0):
    return gaussian_filter(image, sigma=sigma)


def median_smoothing(image, size=3):
    return median_filter(image, size=size)


def morphological_open_close(image, kernel_size=3):
    """opening removes small bright specks; closing fills holes"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def wavelet_denoise(image, wavelet="db1", level=2, threshold_factor=0.04):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, detail = coeffs[0], coeffs[1:]

    # threshold small wavelet coefficients
    new_detail = []
    for (cH, cV, cD) in detail:
        t = threshold_factor * np.max(cH)
        new_detail.append((
            pywt.threshold(cH, t),
            pywt.threshold(cV, t),
            pywt.threshold(cD, t)
        ))

    return pywt.waverec2([cA] + new_detail, wavelet)


def fft_denoise(image, cutoff=0.1):
    """Low-pass filter in frequency domain"""
    f = np.fft.fftshift(np.fft.fft2(image))

    h, w = image.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    radius = np.sqrt(x*x + y*y)
    mask = radius < cutoff * min(h, w)

    f_filtered = f * mask
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    return img_back


# -------------------------------------------
# Dispatch function
# -------------------------------------------
def denoise(image, method="gaussian", **params):
    if method == "threshold":
        return threshold_filter(image, params.get("threshold", 10))

    elif method == "gaussian":
        return gaussian_smoothing(image, sigma=params.get("sigma", 1.0))

    elif method == "median":
        return median_smoothing(image, size=params.get("size", 3))

    elif method == "morphological":
        return morphological_open_close(image, kernel_size=params.get("kernel_size", 3))

    elif method == "wavelet":
        return wavelet_denoise(
            image,
            wavelet=params.get("wavelet", "db2"),
            level=params.get("level", 2),
            threshold_factor=params.get("threshold_factor", 0.04)
        )

    elif method == "fft":
        return fft_denoise(image, cutoff=params.get("cutoff", 0.08))

    else:
        raise ValueError(f"Unknown method: {method}")


# -------------------------------------------
# Method comparison utility
# -------------------------------------------
def compare_methods(image):
    methods = ["threshold", "gaussian", "median", "morphological", "wavelet", "fft"]
    results = {}

    for m in methods:
        den = denoise(image, method=m)
        metrics = compute_metrics(image, den)
        results[m] = {
            "image": den,
            "metrics": metrics
        }
        print(f"[{m}]  PSNR={metrics['PSNR']:.3f}  SSIM={metrics['SSIM']:.3f}  SNR={metrics['SNR']:.3f}")

    return results

