import argparse
import datetime as dt
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

try:
    from src.integrator.integrate import FastSitnikovSimulation
except Exception:
    class FastSitnikovSimulation:
        def __init__(
            self,
            e,
            n_focal_samples=4096,
            rtol=1e-7,
            atol=1e-10,
            max_step=np.inf,
            solver_method="RK45",
            auto_solver_trials=4,
            phi_time_window=20.0 * np.pi,
            focal_interp="linear",
        ):
            if not (0.0 <= e < 1.0):
                raise ValueError("e must satisfy 0 <= e < 1")
            if n_focal_samples < 16:
                raise ValueError("n_focal_samples must be at least 16")
            if focal_interp not in ("linear", "cubic"):
                raise ValueError("focal_interp must be 'linear' or 'cubic'")

            self.e = float(e)
            self.period = 2.0 * np.pi
            self.a = 0.5
            self.rtol = float(rtol)
            self.atol = float(atol)
            self.max_step = float(max_step)
            self.phi_time_window = float(phi_time_window)
            self.focal_interp = focal_interp

            self.r_min = 0.5 * (1.0 - self.e)
            self.r_min_sq = self.r_min * self.r_min

            self._n_focal_samples = int(n_focal_samples)
            self._t_grid = np.linspace(0.0, self.period, self._n_focal_samples, endpoint=False)
            self.focal_distance_array = self._build_focal_distance_array(self._t_grid)

            self._dt_grid = self.period / self._n_focal_samples
            self._inv_dt_grid = 1.0 / self._dt_grid
            self._r_periodic = np.concatenate((self.focal_distance_array, [self.focal_distance_array[0]]))
            self._t_periodic = np.linspace(0.0, self.period, self._n_focal_samples + 1)

            self._focal_spline = None
            if self.focal_interp == "cubic":
                self._focal_spline = CubicSpline(self._t_periodic, self._r_periodic, bc_type="periodic")

            self.solver_method = solver_method

        def _solve_kepler(self, M, tol=1e-14, max_iter=16):
            E = np.array(M, dtype=float, copy=True)
            for _ in range(max_iter):
                f = E - self.e * np.sin(E) - M
                fp = 1.0 - self.e * np.cos(E)
                dE = f / fp
                E -= dE
                if np.max(np.abs(dE)) < tol:
                    break
            return E

        def _build_focal_distance_array(self, t_grid):
            M = np.mod(t_grid, self.period)
            E = self._solve_kepler(M)
            return self.a * (1.0 - self.e * np.cos(E))

        def _focal_distance_scalar(self, t):
            tau = t % self.period
            if self.focal_interp == "cubic":
                return float(self._focal_spline(tau))
            u = tau * self._inv_dt_grid
            i = int(u)
            frac = u - i
            r0 = self._r_periodic[i]
            r1 = self._r_periodic[i + 1]
            return r0 + frac * (r1 - r0)

        def _rhs(self, t, y):
            z, vz = y
            r = self._focal_distance_scalar(t)
            denom = (z * z + r * r) ** 1.5
            return (vz, -z / denom)

        def stroboscopic(self, z, v):
            sol = solve_ivp(
                self._rhs,
                (0.0, self.period),
                (float(z), float(v)),
                method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            if not sol.success:
                raise RuntimeError(f"stroboscopic integration failed: {sol.message}")
            return float(sol.y[0, -1]), float(sol.y[1, -1])


@dataclass
class ExperimentConfig:
    e: float = 0.57
    n0: int = 10_000
    z0: float = 6.0
    vmin: float = 0.0
    vmax: float = 0.15
    max_rev: int = 160
    escape_abs_z: float = 10.0
    rev_mid: int = 9
    rev_final: int = 16
    checkpoint_every_chunks: int = 1
    chunk_size: int = 2_000
    calibrate_chunk_target_seconds: float = 15.0 * 60.0
    calibrate_max_samples: int = 2_000
    rtol: float = 1e-7
    atol: float = 1e-10
    solver_method: str = "RK45"
    focal_interp: str = "linear"


@dataclass
class OptimizationProfile:
    name: str
    rtol: float
    atol: float
    solver_method: str
    focal_interp: str


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "Unknown"


def is_repo_dirty() -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode("ascii").strip()
        return status != ""
    except Exception:
        return False


def ensure_dirs(experiment_root: Path) -> Dict[str, Path]:
    data_dir = experiment_root / "data"
    plots_dir = experiment_root / "plots"
    runs_dir = data_dir / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return {"data": data_dir, "plots": plots_dir, "runs": runs_dir}


def generate_initial_velocities(cfg: ExperimentConfig) -> np.ndarray:
    return np.linspace(cfg.vmin, cfg.vmax, cfg.n0, dtype=float)


def iter_chunks(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end


def estimate_chunk_size(cfg: ExperimentConfig, seconds_per_sample: float) -> int:
    if seconds_per_sample <= 0.0:
        return cfg.chunk_size
    est = int(cfg.calibrate_chunk_target_seconds / seconds_per_sample)
    # Keep chunk sizes in a practical range for memory and checkpoint cadence.
    return max(500, min(est, 300_000))


def run_single_trajectory(sim: FastSitnikovSimulation, cfg: ExperimentConfig, v0: float):
    z = float(cfg.z0)
    v = float(v0)
    z_mid = np.nan
    v_mid = np.nan
    z_final = np.nan
    v_final = np.nan

    escaped_rev = cfg.max_rev + 1
    for rev in range(1, cfg.max_rev + 1):
        z, v = sim.stroboscopic(z, v)
        if rev == cfg.rev_mid:
            z_mid, v_mid = z, v
        if rev == cfg.rev_final:
            z_final, v_final = z, v
        if abs(z) > cfg.escape_abs_z:
            escaped_rev = rev
            break

    return escaped_rev, (cfg.z0, v0), (z_mid, v_mid), (z_final, v_final)


def process_chunk(sim: FastSitnikovSimulation, cfg: ExperimentConfig, v_chunk: np.ndarray):
    escape_count = np.zeros(cfg.max_rev + 1, dtype=np.int64)
    escape_rev = np.full(v_chunk.shape[0], cfg.max_rev + 1, dtype=np.int32)

    survivors_initial: List[Tuple[float, float]] = []
    survivors_mid: List[Tuple[float, float]] = []
    survivors_final: List[Tuple[float, float]] = []

    for i, v0 in enumerate(v_chunk):
        rev_out, initial_pt, mid_pt, final_pt = run_single_trajectory(sim, cfg, float(v0))
        escape_rev[i] = rev_out
        if rev_out <= cfg.max_rev:
            escape_count[rev_out] += 1

        if rev_out > cfg.rev_final:
            survivors_initial.append(initial_pt)
            survivors_mid.append(mid_pt)
            survivors_final.append(final_pt)

    def as_array(points: List[Tuple[float, float]]) -> np.ndarray:
        if not points:
            return np.empty((0, 2), dtype=float)
        return np.asarray(points, dtype=float)

    return {
        "escape_count": escape_count,
        "escape_rev": escape_rev,
        "initial_points": as_array(survivors_initial),
        "intermediate_points": as_array(survivors_mid),
        "final_points": as_array(survivors_final),
    }


def _merge_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return b.copy()
    if b.size == 0:
        return a
    return np.vstack((a, b))


def save_chunk_result(
    run_dir: Path,
    chunk_id: int,
    start: int,
    end: int,
    v_chunk: np.ndarray,
    chunk_result: Dict[str, np.ndarray],
    elapsed_seconds: float,
):
    out_file = run_dir / f"chunk_{chunk_id:06d}.npz"
    np.savez_compressed(
        out_file,
        chunk_id=chunk_id,
        start_index=start,
        end_index=end,
        elapsed_seconds=elapsed_seconds,
        v_chunk=v_chunk,
        escape_rev=chunk_result["escape_rev"],
        escape_count=chunk_result["escape_count"],
        initial_points=chunk_result["initial_points"],
        intermediate_points=chunk_result["intermediate_points"],
        final_points=chunk_result["final_points"],
    )


def write_manifest(run_dir: Path, manifest: dict):
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_rolling_results(run_dir: Path, agg: Dict[str, np.ndarray]):
    np.savez_compressed(
        run_dir / "rolling_results.npz",
        escape_count=agg["escape_count"],
        initial_points=agg["initial_points"],
        intermediate_points=agg["intermediate_points"],
        final_points=agg["final_points"],
    )


def _calibrate_chunk_size(sim: FastSitnikovSimulation, cfg: ExperimentConfig, v_all: np.ndarray) -> int:
    n_probe = min(cfg.calibrate_max_samples, v_all.shape[0])
    if n_probe < 50:
        return cfg.chunk_size

    probe_cfg = ExperimentConfig(**asdict(cfg))
    probe_cfg.n0 = n_probe
    probe_v = v_all[:n_probe]

    t0 = dt.datetime.now()
    _ = process_chunk(sim, probe_cfg, probe_v)
    dt_seconds = (dt.datetime.now() - t0).total_seconds()
    if dt_seconds <= 0:
        return cfg.chunk_size

    seconds_per_sample = dt_seconds / n_probe
    return estimate_chunk_size(cfg, seconds_per_sample)


def create_sim(cfg: ExperimentConfig) -> FastSitnikovSimulation:
    return FastSitnikovSimulation(
        e=cfg.e,
        rtol=cfg.rtol,
        atol=cfg.atol,
        solver_method=cfg.solver_method,
        focal_interp=cfg.focal_interp,
    )


def _process_chunk_payload(payload):
    cfg_dict, v_chunk = payload
    cfg = ExperimentConfig(**cfg_dict)
    sim = create_sim(cfg)
    return process_chunk(sim, cfg, v_chunk)


def run_experiment(
    cfg: ExperimentConfig,
    run_label: Optional[str] = None,
    resume: bool = False,
    calibrate_chunk_size: bool = True,
    parallel_workers: Optional[int] = None,
) -> Path:
    experiment_root = Path(__file__).resolve().parent
    dirs = ensure_dirs(experiment_root)

    if run_label is None:
        run_label = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = dirs["runs"] / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    v_all = generate_initial_velocities(cfg)
    sim = create_sim(cfg)

    if calibrate_chunk_size and not resume:
        cfg.chunk_size = _calibrate_chunk_size(sim, cfg, v_all)

    total_chunks = int(np.ceil(cfg.n0 / cfg.chunk_size))

    manifest_path = run_dir / "manifest.json"
    if resume and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        agg_npz = np.load(run_dir / "rolling_results.npz")
        agg = {
            "escape_count": agg_npz["escape_count"],
            "initial_points": agg_npz["initial_points"],
            "intermediate_points": agg_npz["intermediate_points"],
            "final_points": agg_npz["final_points"],
        }
        completed_chunks = set(manifest.get("completed_chunks", []))
    else:
        manifest = {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": get_git_revision_hash(),
            "repo_is_dirty": is_repo_dirty(),
            "params": asdict(cfg),
            "completed_chunks": [],
            "total_chunks": total_chunks,
            "status": "running",
            "notes": "Escape at (k,k+1] is counted at k+1."
        }
        agg = {
            "escape_count": np.zeros(cfg.max_rev + 1, dtype=np.int64),
            "initial_points": np.empty((0, 2), dtype=float),
            "intermediate_points": np.empty((0, 2), dtype=float),
            "final_points": np.empty((0, 2), dtype=float),
        }
        completed_chunks = set()
        write_manifest(run_dir, manifest)
        write_rolling_results(run_dir, agg)

    chunks = [
        (chunk_id, start, end)
        for chunk_id, (start, end) in enumerate(iter_chunks(cfg.n0, cfg.chunk_size))
        if chunk_id not in completed_chunks
    ]

    if parallel_workers is not None and parallel_workers > 1 and len(chunks) > 1:
        cfg_payload = asdict(cfg)
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {}
            for chunk_id, start, end in chunks:
                v_chunk = v_all[start:end]
                futures[executor.submit(_process_chunk_payload, (cfg_payload, v_chunk))] = (
                    chunk_id,
                    start,
                    end,
                    v_chunk,
                )

            for future in as_completed(futures):
                chunk_id, start, end, v_chunk = futures[future]
                t0 = dt.datetime.now()
                chunk_result = future.result()
                elapsed = (dt.datetime.now() - t0).total_seconds()

                agg["escape_count"] += chunk_result["escape_count"]
                agg["initial_points"] = _merge_np(agg["initial_points"], chunk_result["initial_points"])
                agg["intermediate_points"] = _merge_np(agg["intermediate_points"], chunk_result["intermediate_points"])
                agg["final_points"] = _merge_np(agg["final_points"], chunk_result["final_points"])

                save_chunk_result(run_dir, chunk_id, start, end, v_chunk, chunk_result, elapsed)
                manifest["completed_chunks"].append(chunk_id)

                if (chunk_id + 1) % cfg.checkpoint_every_chunks == 0:
                    write_rolling_results(run_dir, agg)
                    write_manifest(run_dir, manifest)
    else:
        for chunk_id, start, end in chunks:
            v_chunk = v_all[start:end]
            t0 = dt.datetime.now()
            chunk_result = process_chunk(sim, cfg, v_chunk)
            elapsed = (dt.datetime.now() - t0).total_seconds()

            agg["escape_count"] += chunk_result["escape_count"]
            agg["initial_points"] = _merge_np(agg["initial_points"], chunk_result["initial_points"])
            agg["intermediate_points"] = _merge_np(agg["intermediate_points"], chunk_result["intermediate_points"])
            agg["final_points"] = _merge_np(agg["final_points"], chunk_result["final_points"])

            save_chunk_result(run_dir, chunk_id, start, end, v_chunk, chunk_result, elapsed)
            manifest["completed_chunks"].append(chunk_id)

            if (chunk_id + 1) % cfg.checkpoint_every_chunks == 0:
                write_rolling_results(run_dir, agg)
                write_manifest(run_dir, manifest)

    write_rolling_results(run_dir, agg)
    manifest["status"] = "completed"
    manifest["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    write_manifest(run_dir, manifest)

    return run_dir


def load_results(run_dir: Path) -> Dict[str, np.ndarray]:
    npz = np.load(Path(run_dir) / "rolling_results.npz")
    return {
        "escape_count": npz["escape_count"],
        "initial_points": npz["initial_points"],
        "intermediate_points": npz["intermediate_points"],
        "final_points": npz["final_points"],
    }


def survivors_curve_from_escape_count(escape_count: np.ndarray, n0: int) -> np.ndarray:
    escaped_cumulative = np.cumsum(escape_count)
    return n0 - escaped_cumulative


def _parse_args():
    parser = argparse.ArgumentParser(description="Chaotic saddle reproduction runner")
    parser.add_argument("--n0", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--solver", type=str, default="RK45")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = ExperimentConfig(
        n0=args.n0,
        chunk_size=args.chunk_size,
        checkpoint_every_chunks=args.checkpoint_every,
        rtol=args.rtol,
        atol=args.atol,
        solver_method=args.solver,
    )

    run_dir = run_experiment(
        cfg,
        run_label=args.label,
        resume=args.resume,
        calibrate_chunk_size=(not args.no_calibrate),
        parallel_workers=args.workers,
    )
    print(f"Run completed. Artifacts at: {run_dir}")


if __name__ == "__main__":
    main()
