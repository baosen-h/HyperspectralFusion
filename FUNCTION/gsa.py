from __future__ import annotations

import numpy as np

from FUNCTION.common import *  # noqa: F403


def _gsa_single(hs: np.ndarray, pan: np.ndarray) -> np.ndarray:
    hs = to_hwc(hs)
    pan = np.squeeze(pan).astype(np.float64)
    ratio = pan.shape[0] // hs.shape[0]
    image_lr = upsampling(hs, ratio)
    image_lr0 = image_lr - image_lr.mean(axis=(0, 1), keepdims=True)
    image_lr_lp0 = hs - hs.mean(axis=(0, 1), keepdims=True)
    image_hr0 = imresize_cube(pan - pan.mean(), hs.shape[0], hs.shape[1], order=1)[:, :, 0]
    alpha = estimation_alpha(np.dstack([image_lr_lp0, np.ones(hs.shape[:2])]), image_hr0)
    i_data = np.dstack([image_lr0, np.ones(image_lr.shape[:2])])
    intensity = np.sum(i_data * alpha, axis=2)
    i0 = intensity - intensity.mean()

    var_i = np.var(i0.reshape(-1), ddof=1)
    gains = np.ones(image_lr.shape[2], dtype=np.float64)
    for band in range(image_lr.shape[2]):
        gains[band] = np.cov(i0.reshape(-1), image_lr0[:, :, band].reshape(-1))[0, 1] / var_i
    fused = image_lr0 + (pan - pan.mean() - i0)[:, :, None] * gains[None, None, :]
    return fused - fused.mean(axis=(0, 1), keepdims=True) + image_lr.mean(axis=(0, 1), keepdims=True)


def _gsa_grouped(hs: np.ndarray, ms: np.ndarray) -> np.ndarray:
    hs = to_hwc(hs)
    ms = to_hwc(ms)
    rows1, cols1, bands1 = ms.shape
    rows2, cols2, bands2 = hs.shape
    low_res_ms = imresize_cube(ms, rows2, cols2, order=1)
    corr = np.zeros((bands1, bands2), dtype=np.float64)
    for i in range(bands1):
        for j in range(bands2):
            corr[i, j] = corrcoef_scalar(hs[:, :, j], low_res_ms[:, :, i])
    indices = np.argmax(corr, axis=0)
    out = np.zeros((rows1, cols1, bands2), dtype=np.float64)
    for band in range(bands1):
        hs_bands = np.where(indices == band)[0]
        if hs_bands.size:
            out[:, :, hs_bands] = _gsa_single(hs[:, :, hs_bands], ms[:, :, band])
    return out

def gsa(hs: np.ndarray, ms: np.ndarray) -> np.ndarray:
    return _gsa_grouped(hs, ms)
