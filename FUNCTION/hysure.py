from __future__ import annotations

import time

import numpy as np
from scipy.linalg import toeplitz

from FUNCTION.common import *  # noqa: F403


def im2mat(image: np.ndarray) -> np.ndarray:
    image = to_hwc(image)
    return image.reshape(-1, image.shape[2], order="F").T


def mat2im(matrix: np.ndarray, rows: int) -> np.ndarray:
    bands, pixels = matrix.shape
    cols = pixels // rows
    return matrix.T.reshape(rows, cols, bands, order="F")


def circular_convolve(x: np.ndarray, fk: np.ndarray, rows: int) -> np.ndarray:
    bands, pixels = x.shape
    cols = pixels // rows
    cube = x.T.reshape(rows, cols, bands, order="F")
    out = np.real(np.fft.ifft2(np.fft.fft2(cube, axes=(0, 1)) * fk[:, :, None], axes=(0, 1)))
    return out.reshape(-1, bands, order="F").T


def upsample_hs(yhim: np.ndarray, factor: int, rows: int, cols: int, shift: int = 0) -> np.ndarray:
    yhim = to_hwc(yhim)
    out = np.zeros((yhim.shape[0] * factor, yhim.shape[1] * factor, yhim.shape[2]), dtype=np.float64)
    out[shift::factor, shift::factor, :] = yhim
    return out[:rows, :cols, :]


def downsample_hs(yhim_up: np.ndarray, factor: int, shift: int = 0) -> np.ndarray:
    yhim_up = to_hwc(yhim_up)
    return yhim_up[shift::factor, shift::factor, :]


def _center_index(length: int) -> int:
    return int(np.floor((length + 1) / 2 + 0.5)) - 1


def _centered_filter(rows: int, cols: int, kernel: np.ndarray, center_offset: int = 0) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.float64)
    h, w = kernel.shape
    center_r = _center_index(rows) - center_offset
    center_c = _center_index(cols) - center_offset
    r0 = center_r - (h - 1) // 2
    c0 = center_c - (w - 1) // 2
    out[r0 : r0 + h, c0 : c0 + w] = kernel
    return np.fft.ifftshift(out)


def _average_filter(rows: int, cols: int, h: int, w: int) -> np.ndarray:
    kernel = np.ones((h, w), dtype=np.float64) / float(h * w)
    return _centered_filter(rows, cols, kernel)


def _difference_matrix(length: int) -> np.ndarray:
    col = np.zeros(length - 1, dtype=np.float64)
    col[0] = 1.0
    row = np.zeros(length, dtype=np.float64)
    row[0] = 1.0
    row[1] = -1.0
    return toeplitz(col, row)


def vector_soft_col_iso(x1: np.ndarray, x2: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    nu = np.sqrt(np.sum(x1**2, axis=0) + np.sum(x2**2, axis=0))
    shrink = np.maximum(0.0, nu - tau) / np.maximum(nu, np.finfo(np.float64).eps)
    return x1 * shrink[None, :], x2 * shrink[None, :]


def _estimate_band_groups(response: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for band in range(response.shape[0]):
        positive = np.where(response[band] > 0)[0]
        if positive.size:
            groups.append(positive)
        else:
            groups.append(np.arange(response.shape[1]))
    return groups


def estimate_sensor_response(
    yhim: np.ndarray,
    ymim: np.ndarray,
    ratio: int,
    response: np.ndarray,
    p: int,
    shift: int,
    blur_center: int,
    lambda_r: float = 1e1,
    lambda_b: float = 1e1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the HySure subspace and spatial blur.

    The final spectral response uses the shared response estimator, so this
    routine uses the current spectral response only to estimate spatial blur.
    """
    yhim = to_hwc(yhim)
    ymim = to_hwc(ymim)
    rows, cols, ms_bands = ymim.shape
    lr_rows, lr_cols, hs_bands = yhim.shape

    fbm = np.fft.fft2(_average_filter(rows, cols, 9, 9))
    ymb = circular_convolve(im2mat(ymim), fbm, rows)
    ymb_down = im2mat(downsample_hs(mat2im(ymb, rows), ratio, shift))

    small_size = 2 * round(4 / ratio) + 1
    fbh = np.fft.fft2(_average_filter(lr_rows, lr_cols, small_size, small_size))
    yh = im2mat(yhim)
    yhb = circular_convolve(yh, fbh, lr_rows)

    groups = _estimate_band_groups(response)
    r_est = np.zeros((ms_bands, hs_bands), dtype=np.float64)
    for band, indices in enumerate(groups):
        n = indices.size
        if n == 1:
            ddt = np.zeros((1, 1), dtype=np.float64)
        else:
            d = _difference_matrix(n)
            ddt = d.T @ d
        lhs = yhb[indices, :] @ yhb[indices, :].T + lambda_r * ddt
        rhs = yhb[indices, :] @ ymb_down[band, :].T
        r_est[band, indices] = np.linalg.solve(lhs, rhs)

    u, _, _ = np.linalg.svd(yh, full_matrices=False)
    v = u[:, :p]
    yhim_up = upsample_hs(yhim, ratio, rows, cols, shift)
    yh_up = v @ v.T @ im2mat(yhim_up)

    hsize_h = 2 * ratio - 1
    hsize_w = 2 * ratio - 1
    dv = _difference_matrix(hsize_h)
    dh = dv.T
    ah = np.kron(dh.T, np.eye(hsize_w))
    av = np.kron(np.eye(hsize_h), dv)
    regularizer = ah.T @ ah + av.T @ av

    mask2d = np.zeros((rows, cols), dtype=bool)
    mask2d[shift::ratio, shift::ratio] = True
    ryh = r_est @ yh_up
    ryhim = mat2im(ryh, rows)
    ymymt = np.zeros((hsize_h * hsize_w, hsize_h * hsize_w), dtype=np.float64)
    rtyhymt = np.zeros(hsize_h * hsize_w, dtype=np.float64)
    half_h = (hsize_h - 1) // 2
    half_w = (hsize_w - 1) // 2
    for band in range(ms_bands):
        image = ymim[:, :, band]
        for row in range(half_h, rows - half_h - 1):
            for col in range(half_w, cols - half_w - 1):
                if not mask2d[row, col]:
                    continue
                patch = image[row - half_h : row + half_h + 1, col - half_w : col + half_w + 1]
                patch_vec = patch.reshape(-1, order="F")
                ymymt += np.outer(patch_vec, patch_vec)
                rtyhymt += ryhim[row, col, band] * patch_vec

    b_vec = np.linalg.solve(ymymt + lambda_b * regularizer, rtyhymt)
    b_kernel = b_vec.reshape(hsize_h, hsize_w, order="F")
    b = _centered_filter(rows, cols, b_kernel, blur_center)
    volume = np.sum(b)
    if abs(volume) < np.finfo(np.float64).eps:
        volume = 1.0
    return v, b / volume


def _hysure(
    hs: np.ndarray,
    ms: np.ndarray,
    p: int = 30,
    iterations: int = 200,
    shift: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """Run HySure fusion with estimated spatial blur and spectral response."""
    hs = np.maximum(to_hwc(hs), 0)
    ms = np.maximum(to_hwc(ms), 0)
    ratio = ms.shape[0] // hs.shape[0]
    max_hs = np.maximum(float(np.max(hs)), np.finfo(np.float64).eps)
    yhim = hs / max_hs
    ymim = ms / max_hs

    response_aug, _ = estimate_response(yhim, ymim)
    for band in range(ymim.shape[2]):
        ymim[:, :, band] = np.maximum(ymim[:, :, band] - response_aug[band, -1], 0)
    response = response_aug[:, :-1]

    p = min(p, yhim.shape[2])
    rows, cols, _ = ymim.shape
    blur_center = (ratio + 1) % 2
    v_denoise, b = estimate_sensor_response(yhim, ymim, ratio, response, p, shift, blur_center)
    fb = np.fft.fft2(b)
    fbc = np.conj(fb)

    dh = np.zeros((rows, cols), dtype=np.float64)
    dh[0, 0] = 1.0
    dh[0, -1] = -1.0
    dv = np.zeros((rows, cols), dtype=np.float64)
    dv[0, 0] = 1.0
    dv[-1, 0] = -1.0
    fdh = np.fft.fft2(dh)
    fdv = np.fft.fft2(dv)
    fdhc = np.conj(fdh)
    fdvc = np.conj(fdv)
    denom = np.abs(fb) ** 2 + np.abs(fdh) ** 2 + np.abs(fdv) ** 2 + 1.0
    ibd_b = fbc / denom
    ibd_ii = 1.0 / denom
    ibd_dh = fdhc / denom
    ibd_dv = fdvc / denom

    mask2d = np.zeros((rows, cols), dtype=np.float64)
    mask2d[shift::ratio, shift::ratio] = 1.0
    yhim_up = upsample_hs(yhim, ratio, rows, cols, shift)
    yh_up = im2mat(yhim_up)
    mask = im2mat(np.repeat(mask2d[:, :, None], p, axis=2))

    yh_samples = yh_up[:, mask2d.reshape(-1, order="F") > 0]
    max_volume = -np.inf
    e = v_denoise
    for seed in range(20):
        candidate, _ = vca(yh_samples, p, seed=seed)
        volume = abs(np.linalg.det(candidate.T @ candidate))
        if volume > max_volume:
            e = candidate
            max_volume = volume

    mu = 0.05
    lambda_phi = 1e-3
    lambda_m = 1.0
    ie = e.T @ e + mu * np.eye(p)
    yyh = e.T @ yh_up
    ire = lambda_m * e.T @ response.T @ response @ e + mu * np.eye(p)
    ym = im2mat(ymim)
    yym = e.T @ response.T @ ym

    x = np.zeros((p, rows * cols), dtype=np.float64)
    v1 = x.copy()
    d1 = x.copy()
    v2 = x.copy()
    d2 = x.copy()
    v3 = x.copy()
    d3 = x.copy()
    v4 = x.copy()
    d4 = x.copy()

    start_time = time.perf_counter()
    for iteration in range(1, iterations + 1):
        x = (
            circular_convolve(v1 + d1, ibd_b, rows)
            + circular_convolve(v2 + d2, ibd_ii, rows)
            + circular_convolve(v3 + d3, ibd_dh, rows)
            + circular_convolve(v4 + d4, ibd_dv, rows)
        )
        nu1 = circular_convolve(x, fb, rows) - d1
        v1_candidate = np.linalg.solve(ie, yyh + mu * nu1)
        v1 = v1_candidate * mask + nu1 * (1.0 - mask)

        nu2 = x - d2
        v2 = np.linalg.solve(ire, lambda_m * yym + mu * nu2)

        nu3 = circular_convolve(x, fdh, rows) - d3
        nu4 = circular_convolve(x, fdv, rows) - d4
        v3, v4 = vector_soft_col_iso(nu3, nu4, lambda_phi / mu)

        d1 = -nu1 + v1
        d2 = -nu2 + v2
        d3 = -nu3 + v3
        d4 = -nu4 + v4
        if verbose and (iteration == 1 or iteration % 10 == 0 or iteration == iterations):
            elapsed = time.perf_counter() - start_time
            avg = elapsed / iteration
            remaining = avg * (iterations - iteration)
            print(
                f"HySure iteration {iteration}/{iterations} | "
                f"elapsed {elapsed:.1f}s | eta {remaining:.1f}s",
                flush=True,
            )

    z = e @ x
    z = v_denoise @ v_denoise.T @ z
    return mat2im(z, rows) * max_hs

def hysure(hs: np.ndarray, ms: np.ndarray, iterations: int = 200) -> np.ndarray:
    return _hysure(hs, ms, iterations=iterations, shift=1)
