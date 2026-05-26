from __future__ import annotations

import numpy as np

from FUNCTION.common import *  # noqa: F403


def _mtf_glp(hs_band: np.ndarray, pan: np.ndarray, ratio: int, start_pos: tuple[int, int]) -> np.ndarray:
    hs_band = to_hwc(hs_band)
    pan = np.squeeze(pan).astype(np.float64)
    hsu = interp23tap_general(hs_band, ratio, start_pos)
    image_hr = np.repeat(pan[:, :, None], hsu.shape[2], axis=2)
    low = gaussian_down_sample(image_hr, ratio)
    pan_lp = interp23tap_general(low, ratio, start_pos)
    cov = np.cov(hsu.reshape(-1), pan_lp.reshape(-1))
    scaling = cov[0, 1] / cov[1, 1]
    return hsu + scaling * (image_hr - pan_lp)


def _mtf_glp_wrapper(hs: np.ndarray, ms: np.ndarray, mode: int = 1) -> np.ndarray:
    hs = to_hwc(hs)
    ms = to_hwc(ms)
    rows1, cols1, bands1 = ms.shape
    rows2, cols2, bands2 = hs.shape
    ratio = rows1 // rows2
    low_res_ms = imresize_cube(ms, rows2, cols2, order=1)
    a = np.column_stack([low_res_ms.reshape(rows2 * cols2, bands1), np.ones(rows2 * cols2)])

    if mode == 1:
        corr = np.zeros((bands1, bands2), dtype=np.float64)
        for i in range(bands1):
            for j in range(bands2):
                corr[i, j] = corrcoef_scalar(hs[:, :, j], low_res_ms[:, :, i])
        indices = np.argmax(corr, axis=0)
        x = None
    elif mode == 2:
        x = np.linalg.lstsq(a, hs.reshape(rows2 * cols2, bands2), rcond=None)[0]
        indices = None
    elif mode == 3:
        x = nls_coef(hs.reshape(rows2 * cols2, bands2), a)
        indices = None
    else:
        raise ValueError("mode must be 1, 2, or 3")

    a2 = np.column_stack([ms.reshape(rows1 * cols1, bands1), np.ones(rows1 * cols1)])
    pos = (ratio // 2, ratio // 2) if ratio % 2 == 0 else (round(ratio / 2), round(ratio / 2))
    out = np.zeros((rows1, cols1, bands2), dtype=np.float64)
    for j in range(bands2):
        if mode == 1:
            pan = ms[:, :, int(indices[j])]
        else:
            pan = (a2 @ x[:, j]).reshape(rows1, cols1)
        if ratio % 2 == 0:
            out[:, :, j] = 0.5 * (
                _mtf_glp(hs[:, :, j], pan, ratio, pos)[:, :, 0]
                + _mtf_glp(hs[:, :, j], pan, ratio, (pos[0] + 1, pos[1] + 1))[:, :, 0]
            )
        else:
            out[:, :, j] = _mtf_glp(hs[:, :, j], pan, ratio, pos)[:, :, 0]
    return out

def glp(hs: np.ndarray, ms: np.ndarray, mode: int = 1) -> np.ndarray:
    return _mtf_glp_wrapper(hs, ms, mode)


def mtf_glp(hs: np.ndarray, ms: np.ndarray, mode: int = 1) -> np.ndarray:
    return glp(hs, ms, mode)
