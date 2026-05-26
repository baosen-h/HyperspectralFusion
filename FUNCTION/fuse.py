from __future__ import annotations

import numpy as np

from FUNCTION.common import *  # noqa: F403


def _fuse(
    hs: np.ndarray,
    ms: np.ndarray,
    iterations: int = 0,
    spectral_weight: float = 0.35,
    detail_weight: float = 0.90,
) -> np.ndarray:
    """Run fast fusion with spectral and spatial consistency updates."""
    hs = np.maximum(to_hwc(hs), 0)
    ms = np.maximum(to_hwc(ms), 0)
    rows1, cols1, _ = ms.shape
    rows2, cols2, bands2 = hs.shape
    ratio = rows1 // rows2
    scaling = ratio

    response_aug, _ = estimate_response(hs, ms)
    ms_corr = ms.copy()
    for band in range(ms.shape[2]):
        ms_corr[:, :, band] = np.maximum(ms_corr[:, :, band] - response_aug[band, -1], 0)
    response = response_aug[:, :-1]

    hsu = imresize_cube(hs, rows1, cols1, order=3)
    y_m = ms_corr.reshape(-1, ms_corr.shape[2], order="F")
    pred_m = hsu.reshape(-1, bands2, order="F") @ response.T
    spectral_correction = (y_m - pred_m) @ np.linalg.pinv(response).T
    spectral_correction = spectral_correction.reshape(rows1, cols1, bands2, order="F")

    high_synth, low_synth = synthetic_high_images(hs, ms_corr)
    detail = high_synth - imresize_cube(low_synth, rows1, cols1, order=3)

    fused = hsu + spectral_weight * spectral_correction + detail_weight * detail
    if iterations > 0:
        start_pos = (round(ratio / 2), round(ratio / 2))
        kernel_size = round(ratio / 2) * 2 + 1
        sigma = np.sqrt(1.0 / (2.0 * 2.7725887 / (ratio**2)))
        blur_kernel = gaussian_kernel(kernel_size, sigma)
        for _ in range(iterations):
            low = conv_downsample_fast(fused, ratio, blur_kernel, start_pos)
            fused -= 0.05 * imresize_cube(low - hs, rows1, cols1, order=3)

    mean_fused = np.maximum(fused.mean(axis=(0, 1), keepdims=True), np.finfo(np.float64).eps)
    mean_hs = hsu.mean(axis=(0, 1), keepdims=True)
    return np.maximum(fused * (mean_hs / mean_fused), 0)

def fuse(hs: np.ndarray, ms: np.ndarray, iterations: int = 0) -> np.ndarray:
    return _fuse(hs, ms, iterations=iterations)
