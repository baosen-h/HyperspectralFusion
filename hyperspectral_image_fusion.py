from __future__ import annotations

import argparse
from pathlib import Path

from scipy.io import savemat

from FUNCTION.cnmf import cnmf
from FUNCTION.common import load_mat_array, to_hwc
from FUNCTION.fuse import fuse
from FUNCTION.glp import glp, mtf_glp
from FUNCTION.gsa import gsa
from FUNCTION.hysure import hysure
from FUNCTION.mapsmm import mapsmm
from FUNCTION.sfim import sfim


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "DATA" / "OUTPUT"
METHODS = ["sfim", "mapsmm", "gsa", "cnmf", "glp", "fuse", "hysure"]


def run_method(
    method: str,
    hs,
    ms,
    mode: int | None = None,
    iterations: int | None = None,
):
    method = method.lower()
    if method == "sfim":
        return sfim(hs, ms, 3 if mode is None else mode)
    if method == "mapsmm":
        return mapsmm(hs, ms)
    if method == "gsa":
        return gsa(hs, ms)
    if method == "cnmf":
        return cnmf(hs, ms)
    if method in {"glp", "mtf_glp", "mtf-glp"}:
        return glp(hs, ms, 1 if mode is None else mode)
    if method == "fuse":
        return fuse(hs, ms, iterations=0 if iterations is None else iterations)
    if method == "hysure":
        return hysure(hs, ms, iterations=iterations or 200)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traditional HSI-MSI fusion methods.")
    parser.add_argument("--hsi", type=Path, required=True, help="Low-resolution HSI .mat file.")
    parser.add_argument("--msi", type=Path, required=True, help="High-resolution MSI .mat file.")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="sfim")
    parser.add_argument("--hsi-key", default=None, help="Variable name inside HSI .mat file.")
    parser.add_argument("--msi-key", default=None, help="Variable name inside MSI .mat file.")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    hs = to_hwc(load_mat_array(args.hsi, args.hsi_key))
    ms = to_hwc(load_mat_array(args.msi, args.msi_key))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = METHODS if args.method == "all" else [args.method]
    for method in methods:
        fused = run_method(
            method,
            hs,
            ms,
            args.mode,
            iterations=args.iterations,
        )
        output_path = args.output_dir / f"{args.hsi.stem}_{method}.mat"
        savemat(output_path, {"data": fused})
        print(f"{method}: {output_path}")


if __name__ == "__main__":
    main()
