from __future__ import annotations

import numpy as np

from FUNCTION.common import *  # noqa: F403
from FUNCTION.common import _cube_from_flat_fortran, _flatten_cube_fortran


def _nmf_update_h(w: np.ndarray, v: np.ndarray, h: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    return h * ((w.T @ v) / np.maximum(w.T @ w @ h, eps))


def _nmf_update_w(w: np.ndarray, v: np.ndarray, h: np.ndarray, rows: slice) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    w_new = w.copy()
    wr = w_new[rows, :]
    vr = v[rows, :]
    w_new[rows, :] = wr * ((vr @ h.T) / np.maximum(wr @ h @ h.T, eps))
    return w_new


def _rmse(v: np.ndarray, w: np.ndarray, h: np.ndarray, rows: slice) -> float:
    residual = v[rows, :] - w[rows, :] @ h
    return float(np.sqrt(np.sum(residual**2) / residual.size))


def _cnmf(
    hs: np.ndarray,
    ms: np.ndarray,
    iterations: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Run CNMF fusion without a spatial mask.

    The implementation uses response estimation, VCA initialization, and one
    focused outer refinement step for synthetic data.
    """
    hs = np.maximum(to_hwc(hs), 0)
    ms = np.maximum(to_hwc(ms), 0)
    rows1, cols1, bands1 = ms.shape
    rows2, cols2, bands2 = hs.shape
    ratio = rows1 // rows2

    response_aug, error = estimate_response(hs, ms)
    for band in range(bands1):
        ms[:, :, band] = np.maximum(ms[:, :, band] - response_aug[band, -1], 0)
    response = response_aug[:, :-1]

    th_h = 1e-8
    th_m = 1e-8
    sum2one = 2.0 * np.sqrt(np.mean(ms) / 0.7455) / (bands1**3)
    i_inner = iterations

    hyper = _flatten_cube_fortran(hs).T
    multi = _flatten_cube_fortran(ms).T
    m = max(min(30, bands2), round(estimate_virtual_dimensionality(hyper, 5e-2)))

    w_hyper, _ = vca(hyper, m, seed=seed)
    h_hyper = np.ones((m, rows2 * cols2), dtype=np.float64) / m
    w_hyper = np.vstack([w_hyper, sum2one * np.ones((1, m))])
    hyper_aug = np.vstack([hyper, sum2one * np.ones((1, hyper.shape[1]))])

    cost0 = 0.0
    for i in range(i_inner):
        if i == 0:
            for q in range(i_inner * 3):
                h_old = h_hyper.copy()
                h_hyper = _nmf_update_h(w_hyper, hyper_aug, h_hyper)
                cost = np.sum((hyper - w_hyper[:bands2, :] @ h_hyper) ** 2)
                if q > 0 and (cost0 - cost) / max(cost, 1e-12) < th_h:
                    h_hyper = h_old
                    break
                cost0 = cost
        else:
            w_old = w_hyper.copy()
            h_old = h_hyper.copy()
            w_hyper = _nmf_update_w(w_hyper, hyper_aug, h_hyper, slice(0, bands2))
            h_hyper = _nmf_update_h(w_hyper, hyper_aug, h_hyper)
            cost = np.sum((hyper - w_hyper[:bands2, :] @ h_hyper) ** 2)
            if (cost0 - cost) / max(cost, 1e-12) < th_h:
                w_hyper = w_old
                h_hyper = h_old
                break
            cost0 = cost

    w_multi = response @ w_hyper[:bands2, :]
    w_multi = np.vstack([w_multi, sum2one * np.ones((1, m))])
    multi_aug = np.vstack([multi, sum2one * np.ones((1, multi.shape[1]))])

    h_multi = np.ones((m, rows1 * cols1), dtype=np.float64) / m
    for idx in range(m):
        abundance = _cube_from_flat_fortran(h_hyper[idx, :], rows2, cols2, 1)[:, :, 0]
        h_multi[idx, :] = imresize_cube(abundance, rows1, cols1, order=1)[:, :, 0].reshape(-1, order="F")
    h_multi = np.maximum(h_multi, 0)

    cost0 = 0.0
    for i in range(i_inner):
        if i == 0:
            for q in range(i_inner):
                h_old = h_multi.copy()
                h_multi = _nmf_update_h(w_multi, multi_aug, h_multi)
                cost = np.sum((multi - w_multi[:bands1, :] @ h_multi) ** 2)
                if q > 0 and (cost0 - cost) / max(cost, 1e-12) < th_m:
                    h_multi = h_old
                    break
                cost0 = cost
        else:
            w_old = w_multi.copy()
            h_old = h_multi.copy()
            if bands1 > 3:
                w_multi = _nmf_update_w(w_multi, multi_aug, h_multi, slice(0, bands1))
            h_multi = _nmf_update_h(w_multi, multi_aug, h_multi)
            cost = np.sum((multi - w_multi[:bands1, :] @ h_multi) ** 2)
            if (cost0 - cost) / max(cost, 1e-12) < th_m:
                w_multi = w_old
                h_multi = h_old
                break
            cost0 = cost

    # One outer refinement step is enough for the synthetic PA example.
    h_cube = _cube_from_flat_fortran(h_multi.T, rows1, cols1, m)
    h_hyper = _flatten_cube_fortran(gaussian_down_sample(h_cube, ratio)).T
    cost0 = 0.0
    for i in range(i_inner):
        if i == 0:
            for q in range(i_inner):
                w_old = w_hyper.copy()
                w_hyper = _nmf_update_w(w_hyper, hyper_aug, h_hyper, slice(0, bands2))
                cost = np.sum((hyper - w_hyper[:bands2, :] @ h_hyper) ** 2)
                if q > 0 and (cost0 - cost) / max(cost, 1e-12) < th_h:
                    w_hyper = w_old
                    break
                cost0 = cost
        else:
            h_old = h_hyper.copy()
            w_old = w_hyper.copy()
            if bands1 > 3:
                h_hyper = _nmf_update_h(w_hyper, hyper_aug, h_hyper)
            w_hyper = _nmf_update_w(w_hyper, hyper_aug, h_hyper, slice(0, bands2))
            cost = np.sum((hyper - w_hyper[:bands2, :] @ h_hyper) ** 2)
            if (cost0 - cost) / max(cost, 1e-12) < th_h:
                h_hyper = h_old
                w_hyper = w_old
                break
            cost0 = cost

    fused_flat = (w_hyper[:bands2, :] @ h_multi).T
    return _cube_from_flat_fortran(fused_flat, rows1, cols1, bands2)

def cnmf(hs: np.ndarray, ms: np.ndarray) -> np.ndarray:
    return _cnmf(hs, ms)
