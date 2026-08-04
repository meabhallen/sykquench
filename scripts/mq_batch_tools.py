#!/usr/bin/env python3
"""Batch tools for Maldacena-Qi / coupled-SYK equilibrium and real-time KBE runs.

Designed to mirror the single-SYK workflow in syk_batch_tools2.py:

    python mq_batch_tools.py eq-one  --J 1.0 --mu 0.001 --beta 20 --dt 0.005 --omega-max 8 --Nw 4097 --tol 1e-10
    python mq_batch_tools.py kbe-one --J-i 1.0 --J-f 1.0 --mu-i 0.001 --mu-f 0.0 --beta 20 --dt 0.005

Equilibrium output saves the full relative-time 2x2 Green's function G^>_ab(t).
KBE output saves the full two-time matrix G^>_ab(t1,t2), shape (Nt,Nt,2,2), by default.
Checkpoints also store the full in-progress G so a requeued/timed-out job can resume.
"""
from __future__ import annotations

from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import os
import traceback
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
import pandas as pd

from mq_checkpoint_utils import (
    solve_equilibrium_mq_matrix_F_checkpointed,
    evolve_mq_kbe_matrix_checkpointed,
)


# ============================================================
# Helper utilities
# ============================================================

def sf(x: Any) -> str:
    """Safe float-ish string for filenames."""
    try:
        return f"{float(x):.12g}".replace("-", "m").replace(".", "p")
    except Exception:
        return str(x).replace("-", "m").replace(".", "p").replace("/", "_")


def param_hash(params: Dict[str, Any], n: int = 8) -> str:
    """Short stable hash of a parameter dictionary."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def atomic_savez(filename: os.PathLike | str, *, compressed: bool = True, **kwargs: Any) -> None:
    """Atomic npz save by writing to a temp file then renaming."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + ".tmp")
    saver = np.savez_compressed if compressed else np.savez
    with open(tmp, "wb") as f:
        saver(f, **kwargs)
    os.replace(tmp, filename)


def checkpoint_path_for(npz_path: os.PathLike | str) -> Path:
    p = Path(npz_path)
    stem = p.name[:-4] if p.name.endswith(".npz") else p.name
    return p.parent / (stem + ".ckpt.npz")


def update_manifest(manifest_path: os.PathLike | str, row: Dict[str, Any]) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates(subset=["filename"], keep="last")
    else:
        out = new
    out.to_csv(manifest_path, index=False)


def load_manifest(path: os.PathLike | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_manifests_recursive(root: os.PathLike | str, manifest_name: str) -> pd.DataFrame:
    """Load manifest_name from root and all subdirectories; useful when each Slurm job has its own OUT_DIR."""
    root = Path(root)
    frames: List[pd.DataFrame] = []
    for p in sorted(root.rglob(manifest_name)):
        try:
            df = pd.read_csv(p)
            if len(df):
                frames.append(df)
        except Exception as e:
            print(f"Warning: could not read manifest {p}: {e}")
    if not frames:
        direct = root / manifest_name
        return load_manifest(direct)
    out = pd.concat(frames, ignore_index=True)
    if "filename" in out.columns:
        out = out.drop_duplicates(subset=["filename"], keep="last")
    return out


def grid_from_dict(d: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(d.keys())
    vals = [list(d[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def _json_safe(meta: Dict[str, Any]) -> str:
    def default(o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(meta, sort_keys=True, default=default)


def _read_npz_scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = z[key]
    try:
        return arr.item()
    except Exception:
        return arr


# ============================================================
# Filenames
# ============================================================

def equilibrium_filename(out_dir: os.PathLike | str, meta: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    h = param_hash(meta)
    return out_dir / (
        f"mq_eq_J_{sf(meta['J'])}_mu_{sf(meta['mu'])}_beta_{sf(meta['beta'])}"
        f"_dt_{sf(meta['dt'])}_om_{sf(meta['omega_max'])}_Nw_{int(meta['Nw'])}"
        f"_tol_{sf(meta['tol'])}_{h}.npz"
    )


def kbe_filename(out_dir: os.PathLike | str, meta: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    h = param_hash(meta)
    return out_dir / (
        f"mq_kbe_J_{sf(meta['J_i'])}_to_{sf(meta['J_f'])}"
        f"_mu_{sf(meta['mu_i'])}_to_{sf(meta['mu_f'])}"
        f"_beta_{sf(meta['beta'])}_dt_{sf(meta['dt'])}"
        f"_tpre_{sf(meta['t_pre'])}_tpost_{sf(meta['t_post'])}"
        f"_ncorr_{int(meta['n_corr'])}_ctol_{sf(meta['corr_tol'])}"
        f"_it_{int(meta['iterations'])}_{h}.npz"
    )


# ============================================================
# Equilibrium wrappers
# ============================================================

def _solve_or_load_mu0_seed(
    *,
    J: float,
    beta: float,
    out_dir: Path,
    dt: float,
    omega_max: float,
    Nw: int,
    t_max: Optional[float],
    max_iter: int,
    seed_tol: float,
    mixing: float,
    eta_ret: float,
    verbose_every: int,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
    project: bool,
) -> tuple[np.ndarray, np.ndarray]:
    seed_meta = {
        "kind": "equilibrium_seed_mu0",
        "J": float(J),
        "mu": 0.0,
        "beta": float(beta),
        "dt": float(dt),
        "omega_max": float(omega_max),
        "Nw": int(Nw),
        "t_max": None if t_max is None else float(t_max),
        "tol": float(seed_tol),
        "max_iter": int(max_iter),
        "mixing": float(mixing),
        "eta_ret": float(eta_ret),
        "project": bool(project),
    }
    seed_file = equilibrium_filename(out_dir, seed_meta)
    seed_ckpt = checkpoint_path_for(seed_file)

    if seed_file.exists() and not overwrite:
        print(f"Loading existing mu=0 seed: {seed_file}")
        z = np.load(seed_file, allow_pickle=True)
        return z["omega"], z["GR_w"]

    print("Solving mu=0 seed for weak-MQ initialisation...")
    omega0, _t0, _Ggt0, GR_w0, _F_t0 = solve_equilibrium_mq_matrix_F_checkpointed(
        J=J,
        mu=0.0,
        beta=beta,
        omega_max=omega_max,
        Nw=Nw,
        t_max=t_max,
        dt=dt,
        max_iter=max_iter,
        tol=seed_tol,
        mixing=mixing,
        eta_ret=eta_ret,
        verbose_every=verbose_every,
        checkpoint_path=seed_ckpt,
        checkpoint_every=checkpoint_every,
        resume=resume,
        final_path=seed_file,
        project_A=project,
        clip_A=project,
        normalize_A=project,
        project_B=project,
        pin_equal_time=project,
        compute_kbe_dab_every=0,
        init_mode="free",
        init_offdiag="zero",
    )
    return omega0, GR_w0


def run_equilibrium_one(
    J: float,
    mu: float,
    beta: float,
    *,
    out_dir: os.PathLike | str = "eq_runs",
    dt: float = 0.1,
    omega_max: float = 8.0,
    Nw: int = 4097,
    t_max: Optional[float] = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
    dab_tol: float = 1e-6,
    require_dab_convergence: bool = False,
    mixing: float = 0.05,
    eta_ret: float = 1e-6,
    verbose_every: int = 25,
    compute_kbe_dab_every: int = 50,
    checkpoint_every: int = 25,
    seed_from_mu0: bool = False,
    seed_tol: Optional[float] = None,
    project: bool = True,
    overwrite: bool = False,
    resume: bool = True,
) -> Optional[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "mq_eq_manifest.csv"

    meta = {
        "kind": "mq_equilibrium",
        "J": float(J),
        "mu": float(mu),
        "beta": float(beta),
        "dt": float(dt),
        "omega_max": float(omega_max),
        "Nw": int(Nw),
        "t_max": None if t_max is None else float(t_max),
        "tol": float(tol),
        "dab_tol": float(dab_tol),
        "require_dab_convergence": bool(require_dab_convergence),
        "max_iter": int(max_iter),
        "mixing": float(mixing),
        "eta_ret": float(eta_ret),
        "compute_kbe_dab_every": int(compute_kbe_dab_every),
        "checkpoint_every": int(checkpoint_every),
        "seed_from_mu0": bool(seed_from_mu0),
        "project": bool(project),
    }
    filename = equilibrium_filename(out_dir, meta)
    checkpoint_path = checkpoint_path_for(filename)

    if filename.exists() and not overwrite:
        print(f"Skipping existing equilibrium file: {filename}")
        return filename

    init_kwargs: Dict[str, Any] = dict(init_mode="free", init_offdiag="zero")
    if seed_from_mu0 and abs(mu) > 0:
        st = float(seed_tol if seed_tol is not None else max(tol, 1e-5))
        omega0, GR_w0 = _solve_or_load_mu0_seed(
            J=J,
            beta=beta,
            out_dir=out_dir,
            dt=dt,
            omega_max=omega_max,
            Nw=Nw,
            t_max=t_max,
            max_iter=max(200, min(max_iter, 2000)),
            seed_tol=st,
            mixing=mixing,
            eta_ret=eta_ret,
            verbose_every=verbose_every,
            checkpoint_every=checkpoint_every,
            resume=resume,
            overwrite=False,
            project=project,
        )
        init_kwargs = dict(
            init_mode="uncoupled_syk",
            init_omega=omega0,
            init_GR_LL=GR_w0[:, 0, 0],
            init_GR_RR=GR_w0[:, 1, 1],
            init_offdiag="linear_mu",
        )

    print("\n============================================================")
    print("Solving MQ equilibrium")
    print(json.dumps(meta, indent=2))
    print("Saving to:", filename)
    print("Checkpoint:", checkpoint_path)
    print("============================================================")

    try:
        solve_equilibrium_mq_matrix_F_checkpointed(
            J=J,
            mu=mu,
            beta=beta,
            omega_max=omega_max,
            Nw=Nw,
            t_max=t_max,
            dt=dt,
            max_iter=max_iter,
            tol=tol,
            dab_tol=dab_tol,
            require_dab_convergence=require_dab_convergence,
            mixing=mixing,
            eta_ret=eta_ret,
            verbose_every=verbose_every,
            compute_kbe_dab_every=compute_kbe_dab_every,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
            final_path=filename,
            project_A=project,
            clip_A=project,
            normalize_A=project,
            project_B=project,
            pin_equal_time=project,
            **init_kwargs,
        )
        row = dict(meta)
        row.update({"filename": str(filename), "status": "ok"})
        update_manifest(manifest_path, row)
        print("Saved:", filename)
        return filename
    except Exception as e:
        traceback.print_exc()
        row = dict(meta)
        row.update({"filename": str(filename), "status": "failed", "error": repr(e)})
        update_manifest(manifest_path, row)
        return None


def run_equilibrium_grid(eq_grid: Dict[str, Iterable[Any]], *, out_dir: os.PathLike | str = "eq_runs", overwrite: bool = False) -> List[Optional[Path]]:
    paths = []
    for p in grid_from_dict(eq_grid):
        paths.append(run_equilibrium_one(out_dir=out_dir, overwrite=overwrite, **p))
    return paths


# ============================================================
# KBE wrappers
# ============================================================

def find_eq_file(
    eq_dir: os.PathLike | str,
    *,
    J: float,
    mu: float,
    beta: float,
    dt: Optional[float] = None,
    omega_max: Optional[float] = None,
    Nw: Optional[int] = None,
    prefer_smallest_dt: bool = True,
) -> Path:
    manifest = load_manifests_recursive(eq_dir, "mq_eq_manifest.csv")
    if manifest.empty:
        raise FileNotFoundError(f"No mq_eq_manifest.csv files found under {eq_dir}")

    good = manifest[
        (manifest["status"] == "ok")
        & np.isclose(manifest["J"].astype(float), float(J))
        & np.isclose(manifest["mu"].astype(float), float(mu))
        & np.isclose(manifest["beta"].astype(float), float(beta))
    ].copy()
    if dt is not None and "dt" in good.columns:
        good = good[np.isclose(good["dt"].astype(float), float(dt))]
    if omega_max is not None and "omega_max" in good.columns:
        good = good[np.isclose(good["omega_max"].astype(float), float(omega_max))]
    if Nw is not None and "Nw" in good.columns:
        good = good[good["Nw"].astype(int) == int(Nw)]

    if len(good) == 0:
        raise FileNotFoundError(
            f"No equilibrium file found for J={J}, mu={mu}, beta={beta}, "
            f"dt={dt}, omega_max={omega_max}, Nw={Nw} under {eq_dir}"
        )
    sort_cols: List[str] = []
    ascending: List[bool] = []
    for col, asc in [("dt", True), ("tol", True), ("Nw", False), ("omega_max", False)]:
        if col in good.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if sort_cols:
        good = good.sort_values(sort_cols, ascending=ascending)
    return Path(good.iloc[0]["filename"])


def run_kbe_one(
    J_i: float,
    J_f: float,
    mu_i: float,
    mu_f: float,
    beta: float,
    *,
    dt: float = 0.1,
    t_pre: Optional[float] = None,
    t_post: Optional[float] = None,
    t_pre_factor: Optional[float] = 2.0,
    t_post_factor: Optional[float] = 1.0,
    n_corr: int = 6,
    corr_tol: float = 1e-10,
    iterations: int = 50,
    eq_dir: os.PathLike | str = "eq_runs",
    out_dir: os.PathLike | str = "kbe_runs",
    eq_file: Optional[os.PathLike | str] = None,
    eq_dt: Optional[float] = None,
    eq_omega_max: Optional[float] = None,
    eq_Nw: Optional[int] = None,
    keep_cross_self_energy: bool = True,
    checkpoint_every: int = 25,
    overwrite: bool = False,
    resume: bool = True,
    compressed: bool = True,
) -> Optional[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "mq_kbe_manifest.csv"

    if t_pre is None:
        if t_pre_factor is None:
            raise ValueError("Need either t_pre or t_pre_factor.")
        t_pre = float(t_pre_factor) * float(beta)
    if t_post is None:
        if t_post_factor is None:
            raise ValueError("Need either t_post or t_post_factor.")
        t_post = float(t_post_factor) * float(beta)

    if eq_file is None:
        eq_file = find_eq_file(
            eq_dir,
            J=J_i,
            mu=mu_i,
            beta=beta,
            dt=eq_dt,
            omega_max=eq_omega_max,
            Nw=eq_Nw,
        )
    eq_file = Path(eq_file)

    meta = {
        "kind": "mq_kbe",
        "J_i": float(J_i),
        "J_f": float(J_f),
        "mu_i": float(mu_i),
        "mu_f": float(mu_f),
        "beta": float(beta),
        "dt": float(dt),
        "t_pre": float(t_pre),
        "t_post": float(t_post),
        "t_pre_factor": None if t_pre_factor is None else float(t_pre_factor),
        "t_post_factor": None if t_post_factor is None else float(t_post_factor),
        "n_corr": int(n_corr),
        "corr_tol": float(corr_tol),
        "iterations": int(iterations),
        "checkpoint_every": int(checkpoint_every),
        "keep_cross_self_energy": bool(keep_cross_self_energy),
        "eq_source": str(eq_file),
        "eq_dt_filter": None if eq_dt is None else float(eq_dt),
        "eq_omega_max_filter": None if eq_omega_max is None else float(eq_omega_max),
        "eq_Nw_filter": None if eq_Nw is None else int(eq_Nw),
        "save_format": "full_two_time_Ggt",
        "Ggt_shape": "(Nt,Nt,2,2)",
        "saved_bosonic_primaries": False,
    }
    filename = kbe_filename(out_dir, meta)
    checkpoint_path = checkpoint_path_for(filename)

    if filename.exists() and not overwrite:
        print(f"Skipping existing KBE file: {filename}")
        return filename

    print("\n============================================================")
    print("Running MQ KBE")
    print(json.dumps(meta, indent=2))
    print("Saving full two-time GF to:", filename)
    print("Checkpoint:", checkpoint_path)
    print("============================================================")

    try:
        z = np.load(eq_file, allow_pickle=True)
        if "t_eq" in z.files:
            t_eq = z["t_eq"]
        elif "t" in z.files:
            t_eq = z["t"]
        else:
            raise KeyError(f"Equilibrium file {eq_file} contains no t_eq/t array")

        if "Ggt_eq" in z.files:
            Ggt_eq = z["Ggt_eq"]
        elif "Ggt" in z.files:
            Ggt_eq = z["Ggt"]
        else:
            raise KeyError(f"Equilibrium file {eq_file} contains no Ggt_eq/Ggt array")

        Nw_loaded = int(_read_npz_scalar(z, "Nw", -1))
        omega_loaded = z["omega"] if "omega" in z.files else None
        GR_w = z["GR_w"] if "GR_w" in z.files else None
        F_t = z["F_t"] if "F_t" in z.files else None

        t_kbe, Ggt = evolve_mq_kbe_matrix_checkpointed(
            t_eq,
            Ggt_eq,
            J_i=J_i,
            J_f=J_f,
            mu_i=mu_i,
            mu_f=mu_f,
            beta=beta,
            t_pre=t_pre,
            t_post=t_post,
            dt=dt,
            n_corr=n_corr,
            corr_tol=corr_tol,
            print_every=iterations,
            keep_cross_self_energy=keep_cross_self_energy,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
            final_path=None,  # Save below so metadata/eq arrays match the batch manifest.
            Nw=Nw_loaded,
            save_full_final_G=False,
        )

        n0 = int(np.argmin(np.abs(t_kbe)))
        t_kbe[n0] = 0.0
        idx = np.arange(len(t_kbe))
        B_LR_equal = -np.real(Ggt[idx, idx, 0, 1])

        payload: Dict[str, Any] = {
            "t_kbe": t_kbe,
            "Ggt": Ggt,
            "n0": np.array(n0),
            "B_LR_equal": B_LR_equal,
            "t_eq": t_eq,
            "Ggt_eq": Ggt_eq,
            "J_i": np.array(J_i),
            "J_f": np.array(J_f),
            "mu_i": np.array(mu_i),
            "mu_f": np.array(mu_f),
            "beta": np.array(beta),
            "dt": np.array(dt),
            "t_pre": np.array(t_pre),
            "t_post": np.array(t_post),
            "Nw": np.array(Nw_loaded),
            "metadata_json": _json_safe(meta),
        }
        if omega_loaded is not None:
            payload["omega"] = omega_loaded
        if GR_w is not None:
            payload["GR_w_eq"] = GR_w
        if F_t is not None:
            payload["F_t_eq"] = F_t

        atomic_savez(filename, compressed=compressed, **payload)

        row = dict(meta)
        row.update({"filename": str(filename), "status": "ok"})
        update_manifest(manifest_path, row)
        print("Saved:", filename)
        return filename
    except Exception as e:
        traceback.print_exc()
        row = dict(meta)
        row.update({"filename": str(filename), "status": "failed", "error": repr(e)})
        update_manifest(manifest_path, row)
        return None


def run_kbe_grid(kbe_grid: Dict[str, Iterable[Any]], *, eq_dir: os.PathLike | str = "eq_runs", out_dir: os.PathLike | str = "kbe_runs", overwrite: bool = False) -> List[Optional[Path]]:
    paths = []
    for p in grid_from_dict(kbe_grid):
        paths.append(run_kbe_one(eq_dir=eq_dir, out_dir=out_dir, overwrite=overwrite, **p))
    return paths


def load_mq_kbe_dataframe(kbe_dir: os.PathLike | str = "kbe_runs") -> pd.DataFrame:
    manifest = load_manifests_recursive(kbe_dir, "mq_kbe_manifest.csv")
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        if row.get("status") != "ok":
            continue
        filename = Path(row["filename"])
        if not filename.exists():
            print("Missing:", filename)
            continue
        z = np.load(filename, allow_pickle=True)
        entry: Dict[str, Any] = dict(row)
        entry.update({
            "filename": str(filename),
            "t_kbe": z["t_kbe"],
            "Ggt": z["Ggt"],
            "n0": int(z["n0"]),
            "B_LR_equal": z["B_LR_equal"] if "B_LR_equal" in z.files else None,
            "t_eq": z["t_eq"] if "t_eq" in z.files else None,
            "Ggt_eq": z["Ggt_eq"] if "Ggt_eq" in z.files else None,
        })
        rows.append(entry)
    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch utilities for MQ/coupled-SYK equilibrium and KBE runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    peq = sub.add_parser("eq-one", help="Run one MQ equilibrium solve.")
    peq.add_argument("--J", type=float, required=True)
    peq.add_argument("--mu", type=float, required=True)
    peq.add_argument("--beta", type=float, required=True)
    peq.add_argument("--out-dir", default="eq_runs")
    peq.add_argument("--dt", type=float, default=0.1)
    peq.add_argument("--omega-max", type=float, default=8.0)
    peq.add_argument("--Nw", type=int, default=4097)
    peq.add_argument("--t-max", type=float, default=None)
    peq.add_argument("--max-iter", type=int, default=2000)
    peq.add_argument("--tol", type=float, default=1e-9)
    peq.add_argument("--dab-tol", type=float, default=1e-6, help="Tolerance for max(d_ab**0.5). Printed always when d_ab is computed; only required if --require-dab-convergence is set.")
    peq.add_argument("--require-dab-convergence", action="store_true", help="Require max(d_ab**0.5) < --dab-tol before marking equilibrium converged.")
    peq.add_argument("--mixing", type=float, default=0.05)
    peq.add_argument("--eta-ret", type=float, default=1e-6)
    peq.add_argument("--verbose-every", type=int, default=25)
    peq.add_argument("--compute-kbe-dab-every", type=int, default=50)
    peq.add_argument("--checkpoint-every", type=int, default=25)
    peq.add_argument("--seed-from-mu0", action="store_true")
    peq.add_argument("--seed-tol", type=float, default=None)
    peq.add_argument("--no-project", action="store_true")
    peq.add_argument("--no-resume", action="store_true")
    peq.add_argument("--overwrite", action="store_true")

    pkbe = sub.add_parser("kbe-one", help="Run one MQ KBE evolution from an equilibrium npz.")
    pkbe.add_argument("--J-i", type=float, required=True)
    pkbe.add_argument("--J-f", type=float, default=None, help="Defaults to J-i")
    pkbe.add_argument("--mu-i", type=float, required=True)
    pkbe.add_argument("--mu-f", type=float, required=True)
    pkbe.add_argument("--beta", type=float, required=True)
    pkbe.add_argument("--dt", type=float, default=0.1)
    pkbe.add_argument("--t-pre", type=float, default=None)
    pkbe.add_argument("--t-post", type=float, default=None)
    pkbe.add_argument("--t-pre-factor", type=float, default=2.0)
    pkbe.add_argument("--t-post-factor", type=float, default=1.0)
    pkbe.add_argument("--n-corr", type=int, default=6)
    pkbe.add_argument("--corr-tol", type=float, default=1e-10)
    pkbe.add_argument("--iterations", type=int, default=50, help="Print every this many post-quench steps.")
    pkbe.add_argument("--eq-dir", default="eq_runs")
    pkbe.add_argument("--out-dir", default="kbe_runs")
    pkbe.add_argument("--eq-file", default=None)
    pkbe.add_argument("--eq-dt", type=float, default=None, help="Optional filter for equilibrium file lookup.")
    pkbe.add_argument("--eq-omega-max", type=float, default=None, help="Optional filter for equilibrium file lookup.")
    pkbe.add_argument("--eq-Nw", type=int, default=None, help="Optional filter for equilibrium file lookup.")
    pkbe.add_argument("--checkpoint-every", type=int, default=25)
    pkbe.add_argument("--drop-cross-self-energy", action="store_true")
    pkbe.add_argument("--uncompressed", action="store_true", help="Use faster/larger np.savez instead of np.savez_compressed.")
    pkbe.add_argument("--no-resume", action="store_true")
    pkbe.add_argument("--overwrite", action="store_true")

    pgrid = sub.add_parser("grid", help="Run eq-grid or kbe-grid from a JSON config file.")
    pgrid.add_argument("--kind", choices=["eq", "kbe"], required=True)
    pgrid.add_argument("--config", required=True, help="JSON file containing a dict of parameter lists.")
    pgrid.add_argument("--eq-dir", default="eq_runs")
    pgrid.add_argument("--out-dir", default=None)
    pgrid.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "eq-one":
        run_equilibrium_one(
            J=args.J,
            mu=args.mu,
            beta=args.beta,
            out_dir=args.out_dir,
            dt=args.dt,
            omega_max=args.omega_max,
            Nw=args.Nw,
            t_max=args.t_max,
            max_iter=args.max_iter,
            tol=args.tol,
            dab_tol=args.dab_tol,
            require_dab_convergence=args.require_dab_convergence,
            mixing=args.mixing,
            eta_ret=args.eta_ret,
            verbose_every=args.verbose_every,
            compute_kbe_dab_every=args.compute_kbe_dab_every,
            checkpoint_every=args.checkpoint_every,
            seed_from_mu0=args.seed_from_mu0,
            seed_tol=args.seed_tol,
            project=not args.no_project,
            overwrite=args.overwrite,
            resume=not args.no_resume,
        )
    elif args.cmd == "kbe-one":
        run_kbe_one(
            J_i=args.J_i,
            J_f=args.J_f if args.J_f is not None else args.J_i,
            mu_i=args.mu_i,
            mu_f=args.mu_f,
            beta=args.beta,
            dt=args.dt,
            t_pre=args.t_pre,
            t_post=args.t_post,
            t_pre_factor=args.t_pre_factor,
            t_post_factor=args.t_post_factor,
            n_corr=args.n_corr,
            corr_tol=args.corr_tol,
            iterations=args.iterations,
            eq_dir=args.eq_dir,
            out_dir=args.out_dir,
            eq_file=args.eq_file,
            eq_dt=args.eq_dt,
            eq_omega_max=args.eq_omega_max,
            eq_Nw=args.eq_Nw,
            keep_cross_self_energy=not args.drop_cross_self_energy,
            checkpoint_every=args.checkpoint_every,
            overwrite=args.overwrite,
            resume=not args.no_resume,
            compressed=not args.uncompressed,
        )
    elif args.cmd == "grid":
        with open(args.config) as f:
            cfg = json.load(f)
        if args.kind == "eq":
            run_equilibrium_grid(cfg, out_dir=args.out_dir or "eq_runs", overwrite=args.overwrite)
        else:
            run_kbe_grid(cfg, eq_dir=args.eq_dir, out_dir=args.out_dir or "kbe_runs", overwrite=args.overwrite)


if __name__ == "__main__":
    main()
