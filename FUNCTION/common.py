from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve, convolve1d, zoom
from scipy.optimize import lsq_linear, nnls
from scipy.signal import firwin
from scipy.stats import norm
from skimage.transform import resize


def load_mat_array(path: Path, key: str | None = None) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.read(8)
    if header == b"\x89HDF\r\n\x1a\n":
        with h5py.File(path, "r") as file:
            arrays = {
                name: np.array(value).transpose()
                for name, value in file.items()
                if hasattr(value, "shape")
                and np.issubdtype(value.dtype, np.number)
                and len(value.shape) in (2, 3)
            }
    else:
        data = loadmat(path)
        arrays = {
            name: value
            for name, value in data.items()
            if not name.startswith("__")
            and isinstance(value, np.ndarray)
            and np.issubdtype(value.dtype, np.number)
            and value.ndim in (2, 3)
        }

    if key:
        if key not in arrays:
            raise KeyError(f"{key!r} was not found in {path.name}")
        return np.asarray(arrays[key], dtype=np.float64)
    if not arrays:
        raise ValueError(f"No numeric 2-D/3-D array found in {path}")
    return np.asarray(max(arrays.values(), key=lambda item: item.size), dtype=np.float64)


def to_hwc(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 2:
        return image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {image.shape}")
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.moveaxis(image, 0, -1)
    return image.astype(np.float64, copy=False)


def imresize_cube(image: np.ndarray, rows: int, cols: int, order: int = 1) -> np.ndarray:
    image = to_hwc(image)
    anti_aliasing = order > 0 and (rows < image.shape[0] or cols < image.shape[1])
    return resize(
        image,
        (rows, cols, image.shape[2]),
        order=order,
        mode="edge",
        anti_aliasing=anti_aliasing,
        preserve_range=True,
    ).astype(np.float64, copy=False)


def corrcoef_scalar(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def nls_coef(y: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve nonnegative least-squares coefficients band by band."""
    p = y.shape[1]
    m = d.shape[1]
    x = np.zeros((m, p), dtype=np.float64)
    for i in range(p):
        x[:, i] = nnls(d, y[:, i])[0]
    return x


def estimation_alpha(i_ms: np.ndarray, i_pan: np.ndarray) -> np.ndarray:
    """Estimate global linear-regression coefficients for GS/GSA."""
    i_ms = to_hwc(i_ms)
    ihc = i_pan.reshape(-1, 1)
    ilrc = i_ms.reshape(i_ms.shape[0] * i_ms.shape[1], i_ms.shape[2])
    return np.linalg.lstsq(ilrc, ihc, rcond=None)[0].reshape(1, 1, -1)


def upsampling(data: np.ndarray, ratio: int) -> np.ndarray:
    """Upsample a cube with bilinear interpolation."""
    data = to_hwc(data)
    return imresize_cube(data, data.shape[0] * ratio, data.shape[1] * ratio, order=1)


def interp23tap_general(data: np.ndarray, ratio: int, start_pos: tuple[int, int] = (1, 1)) -> np.ndarray:
    """Upsample with a separable interpolation filter."""
    data = to_hwc(data)
    rows, cols, bands = data.shape
    coeff = ratio * firwin(45, 1.0 / ratio)
    out = np.zeros((ratio * rows, ratio * cols, bands), dtype=np.float64)
    r0 = start_pos[0] - 1
    c0 = start_pos[1] - 1
    out[r0::ratio, c0::ratio, :] = data
    for band in range(bands):
        tmp = convolve1d(out[:, :, band].T, coeff, axis=1, mode="wrap")
        out[:, :, band] = convolve1d(tmp.T, coeff, axis=1, mode="wrap")
    return out


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def conv_downsample_fast(
    image: np.ndarray, ratio: int, kernel: np.ndarray, start_pos: tuple[int, int]
) -> np.ndarray:
    image = to_hwc(image)
    blurred = np.zeros_like(image, dtype=np.float64)
    for band in range(image.shape[2]):
        blurred[:, :, band] = convolve(image[:, :, band], kernel, mode="wrap")
    return blurred[start_pos[0] - 1 :: ratio, start_pos[1] - 1 :: ratio, :]


def synthetic_high_images(
    hs: np.ndarray, ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate high-resolution HSI bands from MSI by global linear regression."""
    hs = to_hwc(hs)
    ms = to_hwc(ms)
    rows2, cols2, bands2 = hs.shape
    rows1, cols1, _ = ms.shape
    low_ms = imresize_cube(ms, rows2, cols2, order=1)

    a_low = np.column_stack(
        [low_ms.reshape(-1, low_ms.shape[2], order="F"), np.ones(rows2 * cols2)]
    )
    y_low = hs.reshape(-1, bands2, order="F")
    coef, *_ = np.linalg.lstsq(a_low, y_low, rcond=None)

    a_high = np.column_stack(
        [ms.reshape(-1, ms.shape[2], order="F"), np.ones(rows1 * cols1)]
    )
    high = (a_high @ coef).reshape(rows1, cols1, bands2, order="F")
    low = (a_low @ coef).reshape(rows2, cols2, bands2, order="F")
    return high, low


def gaussian_down_sample(data: np.ndarray, ratio: int) -> np.ndarray:
    """Downsample a cube with a Gaussian spatial kernel."""
    data = to_hwc(data)
    rows, cols, bands = data.shape
    hx = rows // ratio
    hy = cols // ratio
    out = np.zeros((hx, hy, bands), dtype=np.float64)
    sigma = ratio / 2.35482

    if ratio % 2 == 0:
        h1 = gaussian_kernel(ratio, sigma)
        h2 = gaussian_kernel(ratio * 2, sigma)
        pad = ratio // 2
    else:
        h1 = gaussian_kernel(ratio, sigma)
        h2 = gaussian_kernel(ratio * 2 - 1, sigma)
        pad = (ratio - 1) // 2

    for x in range(hx):
        for y in range(hy):
            if x == 0 or x == hx - 1 or y == 0 or y == hy - 1:
                block = data[x * ratio : (x + 1) * ratio, y * ratio : (y + 1) * ratio, :]
                kernel = h1
            else:
                block = data[
                    (x + 1) * ratio - ratio - pad : (x + 1) * ratio + pad,
                    (y + 1) * ratio - ratio - pad : (y + 1) * ratio + pad,
                    :,
                ]
                kernel = h2
            out[x, y, :] = np.sum(block * kernel[:, :, None], axis=(0, 1))
    return out


def _flatten_cube_fortran(cube: np.ndarray) -> np.ndarray:
    cube = to_hwc(cube)
    return cube.reshape(-1, cube.shape[2], order="F")


def _cube_from_flat_fortran(flat: np.ndarray, rows: int, cols: int, bands: int) -> np.ndarray:
    return flat.reshape(rows, cols, bands, order="F")


def estimate_virtual_dimensionality(data: np.ndarray, alpha: float = 5e-2) -> int:
    """Estimate the virtual dimensionality of spectral data."""
    bands, pixels = data.shape
    r = (data @ data.T) / pixels
    k = np.cov(data)
    e_r = np.sort(np.linalg.eigvalsh(r))[::-1]
    e_k = np.sort(np.linalg.eigvalsh(k))[::-1]
    diff = e_r - e_k
    variance = np.sqrt(2.0 * (e_r**2 + e_k**2) / pixels)
    tau = -norm.ppf(alpha, loc=np.zeros(bands), scale=variance)
    return int(np.sum(diff > tau))


def vca(data: np.ndarray, p: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Extract endmember candidates with deterministic VCA."""
    bands, pixels = data.shape
    rng = np.random.default_rng(seed)

    mean_r = data.mean(axis=1, keepdims=True)
    centered = data - mean_r
    u_full, _, _ = np.linalg.svd((centered @ centered.T) / pixels, full_matrices=False)
    ud = u_full[:, :p]
    x_p = ud.T @ centered
    p_y = np.sum(data**2) / pixels
    p_x = np.sum(x_p**2) / pixels + float(mean_r.T @ mean_r)
    snr = abs(10.0 * np.log10(max((p_x - p / bands * p_y), 1e-12) / max((p_y - p_x), 1e-12)))
    snr_threshold = 15.0 + 10.0 * np.log(p) + 8.0

    if snr > snr_threshold:
        d = p
        u_full, _, _ = np.linalg.svd((data @ data.T) / pixels, full_matrices=False)
        ud = u_full[:, :d]
        x = ud.T @ data
        u = x.mean(axis=1, keepdims=True)
        denom = np.sum(x * u, axis=0, keepdims=True)
        y = x / np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    else:
        d = p - 1
        u_full, _, _ = np.linalg.svd((centered @ centered.T) / pixels, full_matrices=False)
        ud = u_full[:, :d]
        x = ud.T @ centered
        c = np.sqrt(np.max(np.sum(x**2, axis=0)))
        y = np.vstack([x, c * np.ones((1, pixels))])

    e_u = np.zeros((p, 1))
    e_u[-1, 0] = 1.0
    a = np.zeros((p, p), dtype=np.float64)
    a[:, [0]] = e_u
    indices = np.zeros(p, dtype=int)
    for i in range(p):
        w = rng.random((p, 1))
        f = w - a @ np.linalg.pinv(a) @ w
        f /= np.sqrt(np.sum(f**2))
        v = f.T @ y
        indices[i] = int(np.argmax(np.abs(v)))
        a[:, i] = y[:, indices[i]]

    if snr > snr_threshold:
        endmembers = ud @ x[:, indices]
    else:
        endmembers = ud @ x[:, indices] + mean_r
    return np.maximum(endmembers, np.finfo(np.float64).eps), indices


def estimate_response(hs: np.ndarray, ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate spectral response coefficients without a spatial mask."""
    hs = np.maximum(to_hwc(hs), 0)
    ms = np.maximum(to_hwc(ms), 0)
    rows1, _, bands1 = ms.shape
    rows2, _, bands2 = hs.shape
    ratio = rows1 // rows2
    lr_ms = gaussian_down_sample(ms, ratio)
    mask = np.ones((rows2, hs.shape[1], 1), dtype=np.float64)
    hs_aug = np.dstack([hs, mask])
    a = _flatten_cube_fortran(hs_aug)
    lower = np.zeros(bands2 + 1)
    lower[-1] = -np.inf
    upper = np.full(bands2 + 1, np.inf)
    response = np.zeros((bands1, bands2 + 1), dtype=np.float64)
    error = np.zeros(bands1, dtype=np.float64)
    for band in range(bands1):
        b = lr_ms[:, :, band].reshape(-1, order="F")
        result = lsq_linear(a, b, bounds=(lower, upper), max_iter=500, lsmr_tol="auto")
        x = result.x
        response[band, :] = x
        denom = np.maximum(np.mean(b), np.finfo(np.float64).eps)
        error[band] = np.sqrt(np.mean((b - a @ x) ** 2)) / denom
    return response, error
