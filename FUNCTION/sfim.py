from __future__ import annotations

import numpy as np

from FUNCTION.common import *  # noqa: F403


def _sfim(hs: np.ndarray, ms: np.ndarray, mode: int = 3) -> np.ndarray:
    hs = to_hwc(hs)
    ms = to_hwc(ms)
    rows1, cols1, bands1 = ms.shape
    rows2, cols2, bands2 = hs.shape
    ratio = rows1 // rows2
    low_res_ms = imresize_cube(ms, rows2, cols2, order=1)

    a = np.column_stack([low_res_ms.reshape(rows2 * cols2, bands1, order="F"), np.ones(rows2 * cols2)])
    if mode == 1:
        corr = np.zeros((bands1, bands2), dtype=np.float64)
        for i in range(bands1):
            for j in range(bands2):
                corr[i, j] = corrcoef_scalar(hs[:, :, j], low_res_ms[:, :, i])
        indices = np.argmax(corr, axis=0)
        x = None
    elif mode == 2:
        ymat = hs.reshape(rows2 * cols2, bands2, order="F")
        x = np.linalg.lstsq(a, ymat, rcond=None)[0]
        indices = None
    elif mode == 3:
        x = nls_coef(hs.reshape(rows2 * cols2, bands2, order="F"), a)
        indices = None
    else:
        raise ValueError("mode must be 1, 2, or 3")

    a2 = np.column_stack([ms.reshape(rows1 * cols1, bands1, order="F"), np.ones(rows1 * cols1)])
    out = np.zeros((rows1, cols1, bands2), dtype=np.float64)
    eps = np.finfo(np.float64).eps
    for i in range(bands2):
        tmp1 = imresize_cube(hs[:, :, i], rows1, cols1, order=0)[:, :, 0]
        if mode == 1:
            idx = int(indices[i])
            tmp2 = ms[:, :, idx]
            tmp3 = low_res_ms[:, :, idx]
        else:
            tmp2 = (a2 @ x[:, i]).reshape(rows1, cols1, order="F")
            tmp3 = (a @ x[:, i]).reshape(rows2, cols2, order="F")
        tmp3_up = imresize_cube(tmp3, rows1, cols1, order=0)[:, :, 0]
        out[:, :, i] = tmp2 * tmp1 / np.where(np.abs(tmp3_up) < eps, eps, tmp3_up)
    return out

def sfim(hs: np.ndarray, ms: np.ndarray, mode: int = 3) -> np.ndarray:
    return _sfim(hs, ms, mode)
