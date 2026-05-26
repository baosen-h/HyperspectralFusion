from __future__ import annotations

import numpy as np

from FUNCTION.common import *  # noqa: F403
from FUNCTION.common import _cube_from_flat_fortran, _flatten_cube_fortran


def pca_mapsmm(data: np.ndarray) -> dict[str, np.ndarray]:
    mean = data.mean(axis=0)
    centered = data - mean[None, :]
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return {"pca": centered @ eigvecs, "eig": eigvals, "a": eigvecs, "mean": mean}


def distribute(m: int, c: int, v: int) -> np.ndarray:
    rows: list[list[int]] = []

    def rec(level: int, remaining: int, current: list[int]) -> None:
        level += 1
        if level < v and remaining > 0:
            limit = current[-1] if current else remaining
            for value in range(int(np.ceil(remaining / (v - level + 1))), min(remaining, limit) + 1):
                rec(level, remaining - value, current + [value])
        elif level == v and remaining > 0:
            rows.append(current + [remaining] + [0] * (c - v))
        else:
            rows.append(current + [0] * (c - level + 1))

    rec(0, m, [])
    perms: set[tuple[int, ...]] = set()
    import itertools

    for row in rows:
        perms.update(itertools.permutations(row))
    return np.array(sorted(perms), dtype=np.float64)


def nfinder_iterate(pct: np.ndarray, filt: np.ndarray) -> np.ndarray:
    nb, nr, nc = pct.shape
    nend = nb + 1
    e = np.zeros((nend, nend), dtype=np.float64)
    adds = 0
    for i in range(nr):
        for j in range(nc):
            if adds < nend and filt[i, j] == 1:
                candidate = pct[:, i, j]
                if not any(np.allclose(candidate, e[1:, k]) for k in range(nend)):
                    e[1:, adds] = candidate
                    adds += 1
    e[0, :] = 1.0
    best_vol = abs(np.linalg.det(e))
    changed = True
    while changed:
        changed = False
        for i in range(nr):
            for j in range(nc):
                if filt[i, j] != 1:
                    continue
                for k in range(nend):
                    test = e.copy()
                    test[:, k] = np.r_[1.0, pct[:, i, j]]
                    vol = abs(np.linalg.det(test))
                    if vol > best_vol:
                        e = test
                        best_vol = vol
                        changed = True
    return e[1:, :]


def mixed_statistics(f: np.ndarray, m_pure: np.ndarray, c_pure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nq, npure = f.shape
    nb = m_pure.shape[0]
    covs = np.zeros((nq, nb, nb), dtype=np.float64)
    means = np.zeros((nb, nq), dtype=np.float64)
    for q in range(nq):
        for cls in range(npure):
            covs[q] += (f[q, cls] ** 2) * c_pure[cls]
            means[:, q] += f[q, cls] * m_pure[:, cls]
    return means, covs


def gaussian_distance(hsi: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    nb, nr, nc = hsi.shape
    cov = cov + 1e-8 * np.eye(nb)
    inv_cov = np.linalg.pinv(cov)
    diff = hsi.reshape(nb, -1, order="F") - mean[:, None]
    dist = 0.5 * np.sum(diff * (inv_cov @ diff), axis=0)
    return dist.reshape(nr, nc, order="F")


def gaussian_pdf(hsi: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    nb = hsi.shape[0]
    cov = cov + 1e-8 * np.eye(nb)
    det = max(float(np.linalg.det(cov)), 1e-300)
    const = ((2.0 * np.pi) ** (-nb / 2.0)) * (det ** -0.5)
    pdf = const * np.exp(-gaussian_distance(hsi, mean, cov))
    return np.maximum(pdf, 0)


def log_likelihood(hsi: np.ndarray, prior: np.ndarray, means: np.ndarray, covs: np.ndarray) -> float:
    psum = np.zeros(hsi.shape[1:], dtype=np.float64)
    for q in range(prior.size):
        psum += prior[q] * gaussian_pdf(hsi, means[:, q], covs[q])
    valid = psum != 0
    if not np.any(valid):
        return float("-inf")
    return float(np.mean(np.log(psum[valid])))


def update_posterior(
    hsi: np.ndarray, prior: np.ndarray, means: np.ndarray, covs: np.ndarray, fractions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nb, nr, nc = hsi.shape
    nq = prior.size
    npure = fractions.shape[1]
    post0 = np.zeros((nq, nr, nc), dtype=np.float64)
    for q in range(nq):
        pdf = gaussian_pdf(hsi, means[:, q], covs[q])
        post0[q] = pdf / max(float(np.sum(pdf)), np.finfo(np.float64).eps)

    denom = np.maximum(np.sum(post0 * prior[:, None, None], axis=0), np.finfo(np.float64).eps)
    post = post0 * prior[:, None, None] / denom[None, :, :]
    weighted = np.tensordot(fractions.T, post, axes=(1, 0))
    fmap = weighted / np.maximum(np.sum(weighted, axis=0, keepdims=True), np.finfo(np.float64).eps)

    hsi2 = hsi.reshape(nb, -1, order="F")
    post2 = post.reshape(nq, -1, order="F")
    new_prior = post2.mean(axis=1)
    new_means = np.zeros_like(means)
    new_covs = np.zeros_like(covs)
    for q in range(nq):
        weights = post2[q]
        total = max(float(np.sum(weights)), np.finfo(np.float64).eps)
        new_means[:, q] = (hsi2 * weights[None, :]).sum(axis=1) / total
        new_covs[q] = (hsi2 * weights[None, :]) @ hsi2.T / total + 1e-8 * np.eye(nb)
    return new_prior, new_means, new_covs, fmap


def run_stochastic_mixing_model(data: np.ndarray, pc_matrix: np.ndarray, npure: int = 4) -> dict[str, np.ndarray]:
    npure_max = npure
    nlevels = npure
    nb = npure + 1
    sigma = 0.0
    scale = 1.0
    keep_fraction = 0.98
    niter = 10

    pct = data[:nb, :, :]
    nr, nc = pct.shape[1:]
    hsi_pixels = data.transpose(1, 2, 0).reshape(-1, data.shape[0], order="F")
    source = np.linalg.solve(pc_matrix.T, hsi_pixels.T).T
    source_cube = source.reshape(nr, nc, data.shape[0], order="F").transpose(2, 0, 1)
    m_pure0, _ = vca(source_cube.reshape(data.shape[0], -1, order="F"), npure)
    m_pure = (m_pure0.T @ pc_matrix).T[:nb, :]

    cov_global = np.cov(pct.reshape(nb, -1, order="F"))
    if sigma <= 0:
        c_pure = np.repeat((scale**2 * cov_global)[None, :, :], npure, axis=0)
    else:
        c_pure = np.repeat((sigma**2 * np.eye(nb))[None, :, :], npure, axis=0)

    fractions = distribute(nlevels, npure, npure_max) / nlevels
    nq = fractions.shape[0]
    index = np.arange(nq - 1, nq - npure - 1, -1)
    prior = np.ones(nq, dtype=np.float64) / nq
    means, covs = mixed_statistics(fractions, m_pure, c_pure)
    # Initialize log-likelihood and scatter, then iterate.
    _ = log_likelihood(pct, prior, means, covs)
    for _iter in range(niter):
        prior, means, covs, fmap = update_posterior(pct, prior, means, covs, fractions)
        m_pure = means[:, nq - npure : nq]
        c_pure = covs[nq - npure : nq]

    post = np.zeros((nq, nr, nc), dtype=np.float64)
    for q in range(nq):
        if prior[q] > 0:
            post[q] = prior[q] * gaussian_pdf(pct, means[:, q], covs[q])
    denom = np.maximum(np.sum(post, axis=0), np.finfo(np.float64).eps)
    post /= denom[None, :, :]
    pindex = np.argmax(post, axis=0)
    fmap = fractions[pindex].transpose(2, 0, 1)
    return {"fmap": fmap, "m_pure": m_pure, "C_pure": c_pure, "index": index}


def class_statistics(u: np.ndarray, cmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(cmap)
    nb = u.shape[0]
    nq = int(labels.max()) + 1
    means = np.zeros((nb, nq), dtype=np.float64)
    covs = np.zeros((nq, nb, nb), dtype=np.float64)
    pixels = u.reshape(nb, -1, order="F")
    flat_labels = cmap.reshape(-1, order="F")
    global_cov = np.cov(pixels) + 1e-8 * np.eye(nb)
    global_mean = pixels.mean(axis=1)
    for q in range(nq):
        selected = pixels[:, flat_labels == q]
        if selected.shape[1] >= 2:
            means[:, q] = selected.mean(axis=1)
            covs[q] = np.cov(selected) + 1e-8 * np.eye(nb)
        else:
            means[:, q] = global_mean
            covs[q] = global_cov
    return means, covs


def pan_to_low_resolution(pan: np.ndarray, mi: int, mj: int) -> np.ndarray:
    return gaussian_down_sample(pan[:, :, None], mi)[:, :, 0]


def replicate_hsi_to_high_resolution(hsi: np.ndarray, mj: int, mi: int) -> np.ndarray:
    nk, rows, cols = hsi.shape
    out = np.zeros((nk, rows * mj, cols * mi), dtype=np.float64)
    for band in range(nk):
        out[band] = imresize_cube(hsi[band, :, :], rows * mj, cols * mi, order=1)[:, :, 0]
    return out


def estimate_high_resolution_pixel(
    x: np.ndarray,
    y: np.ndarray,
    fmap: np.ndarray,
    mj_pure: np.ndarray,
    cj_pure: np.ndarray,
    sigy: float,
) -> np.ndarray:
    nq, ni = fmap.shape
    nm = x.size // ni
    nk = y.size
    cn = (sigy**2) * np.eye(nk)
    h = np.tile(np.eye(nk) / ni, (1, ni))
    czz = np.zeros((nk * ni, nk * ni), dtype=np.float64)
    czx = np.zeros((nk * ni, nm * ni), dtype=np.float64)
    cxz = np.zeros((nm * ni, nk * ni), dtype=np.float64)
    cxx = np.zeros((nm * ni, nm * ni), dtype=np.float64)
    mz = np.zeros(nk * ni, dtype=np.float64)
    mx = np.zeros(nm * ni, dtype=np.float64)
    for pix in range(ni):
        minx, maxx = pix * nm, (pix + 1) * nm
        minz, maxz = pix * nk, (pix + 1) * nk
        for q in range(nq):
            f = fmap[q, pix]
            cov = cj_pure[q]
            czz[minz:maxz, minz:maxz] += f * f * cov[:nk, :nk]
            cxz[minx:maxx, minz:maxz] += f * f * cov[nk : nk + nm, :nk]
            czx[minz:maxz, minx:maxx] += f * f * cov[:nk, nk : nk + nm]
            cxx[minx:maxx, minx:maxx] += f * f * cov[nk : nk + nm, nk : nk + nm]
            mz[minz:maxz] += f * mj_pure[:nk, q]
            mx[minx:maxx] += f * mj_pure[nk : nk + nm, q]

    cxx_inv = np.linalg.pinv(cxx + 1e-8 * np.eye(cxx.shape[0]))
    mzgx = mz + czx @ cxx_inv @ (x - mx)
    czgx = czz - czx @ cxx_inv @ cxz
    gain = czgx @ h.T @ np.linalg.pinv(h @ czgx @ h.T + cn)
    z = mzgx + gain @ (y - h @ mzgx)
    return z.reshape(nk, ni, order="F")


def estimate_high_resolution_cube(
    msi: np.ndarray,
    hsi: np.ndarray,
    fmap: np.ndarray,
    m_pure: np.ndarray,
    c_pure: np.ndarray,
    sigy: float,
) -> np.ndarray:
    nm, nj, ni = msi.shape
    nk, njl, nil = hsi.shape
    nq = fmap.shape[0]
    mj = nj // njl
    mi = ni // nil
    hsie = np.zeros((nk, nj, ni), dtype=np.float64)
    fmap_hr = np.zeros((nq, nj, ni), dtype=np.float64)
    for q in range(nq):
        fmap_hr[q] = imresize_cube(fmap[q, :, :], nj, ni, order=1)[:, :, 0]

    u = np.zeros((nk + nm, njl, nil), dtype=np.float64)
    u[:nk] = hsi
    for k in range(nm):
        u[nk + k] = pan_to_low_resolution(msi[k], mj, mi)
    cmap = np.argmax(fmap, axis=0)
    mj_pure, cj_pure = class_statistics(u, cmap)

    for i in range(nil):
        imin, imax = i * mi, (i + 1) * mi
        for j in range(njl):
            jmin, jmax = j * mj, (j + 1) * mj
            x = msi[:, jmin:jmax, imin:imax].reshape(nm * mi * mj, order="F")
            y = hsi[:, j, i].reshape(nk)
            local_f = fmap_hr[:, jmin:jmax, imin:imax].reshape(nq, mi * mj, order="F")
            z = estimate_high_resolution_pixel(x, y, local_f, mj_pure, cj_pure, sigy)
            hsie[:, jmin:jmax, imin:imax] = z.reshape(nk, mj, mi, order="F")
    return hsie


def _mapsmm(hs: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """Run MAP estimation with a stochastic mixing model."""
    hs = to_hwc(hs)
    ms = to_hwc(ms)
    rows2, cols2, _ = hs.shape
    rows1, _, _ = ms.shape
    ratio = rows1 // rows2

    hsi_pixels = _flatten_cube_fortran(hs)
    pc = pca_mapsmm(hsi_pixels)
    hsi_band_first = hs.transpose(2, 0, 1)
    hsi_pc = _cube_from_flat_fortran(pc["pca"], rows2, cols2, pc["pca"].shape[1]).transpose(2, 0, 1)

    npure = 4
    smm = run_stochastic_mixing_model(hsi_pc, pc["a"], npure)
    labels = np.argmax(smm["fmap"], axis=0)
    if np.unique(labels).size < npure:
        smm = run_stochastic_mixing_model(hsi_pc, pc["a"], npure + 1)
        labels = np.argmax(smm["fmap"], axis=0)
        unique_idx = np.unique(labels)
        smm["C_pure"] = smm["C_pure"][unique_idx]
        smm["m_pure"] = smm["m_pure"][:, unique_idx]
        smm["fmap"] = smm["fmap"][unique_idx]
        npure = unique_idx.size

    method = 15
    nkt = npure + 1
    sigma = np.sqrt(pc["eig"][nkt])
    sigy = sigma / np.sqrt(ratio * ratio)
    msi_band_first = ms.transpose(2, 0, 1)
    nk = hsi_pc.shape[0]
    pcie = np.zeros((nk, rows1, ms.shape[1]), dtype=np.float64)
    c_pure = smm["C_pure"][:, :nkt, :nkt]
    m_pure = smm["m_pure"][:nkt, :]
    pcie[:nkt] = estimate_high_resolution_cube(msi_band_first, hsi_pc[:nkt], smm["fmap"], m_pure, c_pure, sigy)
    pcie[nkt:nk] = replicate_hsi_to_high_resolution(hsi_pc[nkt:nk], ratio, ratio)
    pcie_pixels = pcie.transpose(1, 2, 0).reshape(-1, nk, order="F")
    hsi_rec = pcie_pixels @ np.linalg.inv(pc["a"]) + pc["mean"][None, :]
    return _cube_from_flat_fortran(hsi_rec, rows1, ms.shape[1], hs.shape[2])

def mapsmm(hs: np.ndarray, ms: np.ndarray) -> np.ndarray:
    return _mapsmm(hs, ms)
