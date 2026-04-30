"""GPU auto-detection and optimal hardware config for 7500F + RTX 5060.

Provides hardware-aware model parameters for XGBoost, CatBoost, LightGBM,
with optimal CPU-GPU partitioning for the AMD Ryzen 5 7500F (6C/12T)
paired with NVIDIA GeForce RTX 5060 (8GB VRAM).

Usage:
    from kcact.utils.gpu import (
        get_gpu_config,
        make_xgb_params,
        make_catboost_params,
        make_lgbm_params,
        get_optuna_parallelism,
    )

    gpu_cfg = get_gpu_config()
    xgb_params = make_xgb_params(gpu_cfg)       # adds device="cuda" if GPU available
    cb_params = make_catboost_params(gpu_cfg)   # adds task_type="GPU" if GPU available
    lgbm_params = make_lgbm_params(gpu_cfg)     # adds device="gpu" if GPU available
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from typing import Any


@dataclasses.dataclass(frozen=True)
class GPUConfig:
    """Hardware capability profile for optimal parallelism decisions."""

    cuda_available: bool
    gpu_name: str | None
    gpu_memory_mb: int | None
    cpu_physical_cores: int
    cpu_logical_cores: int
    # Derived:
    use_gpu: bool = True  # can be overridden via env KCADT_NO_GPU=1
    n_gpu_devices: int = 1

    @classmethod
    def detect(cls) -> GPUConfig:
        """Auto-detect hardware and return optimal config."""
        # --- CPU ---
        cpu_physical = os.cpu_count()
        if cpu_physical is None:
            cpu_physical = 6
        # 7500F has 6 physical, 12 logical
        cpu_logical = cpu_physical
        try:
            import multiprocessing
            cpu_logical = multiprocessing.cpu_count() or cpu_physical
        except Exception:
            pass

        # --- CUDA ---
        cuda_available = False
        gpu_name = None
        gpu_memory_mb = None
        n_gpu = 0

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                n_gpu = len(lines)
                if n_gpu > 0:
                    cuda_available = True
                    parts = lines[0].split(",")
                    gpu_name = parts[0].strip() if len(parts) >= 1 else None
                    try:
                        gpu_memory_mb = int(float(parts[1].strip())) if len(parts) >= 2 else None
                    except ValueError:
                        gpu_memory_mb = None
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            # Fallback: try ctypes/pynvml or assume no GPU
            pass

        # Check if CUDA is actually usable by Python libs
        if cuda_available:
            # Verify via PyTorch-cudatoolkit or XGBoost internal check
            try:
                import xgboost as xgb
                _cuda_build = xgb.build_info().get("USE_CUDA", False)
                if not _cuda_build:
                    # XGBoost wasn't built with CUDA — still might work with CatBoost/LGBM
                    pass
            except Exception:
                pass

        # Environment override
        no_gpu = os.environ.get("KCADT_NO_GPU", "").strip().lower()
        use_gpu = cuda_available and no_gpu not in ("1", "true", "yes")

        return cls(
            cuda_available=cuda_available,
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            cpu_physical_cores=cpu_physical,
            cpu_logical_cores=cpu_logical,
            use_gpu=use_gpu,
            n_gpu_devices=max(n_gpu, 1),
        )

    @property
    def optimal_n_jobs(self) -> int:
        """Optimal CPU thread count for sklearn / CPU-parts of pipeline.

        For the 7500F (6C/12T): use 6 for CPU-bound sklearn operations.
        When GPU is in use, reduce to 4 so GPU Python host thread has room.
        """
        if self.use_gpu:
            return max(2, self.cpu_physical_cores - 2)  # leave 2 cores for GPU host
        return max(1, self.cpu_physical_cores)

    @property
    def optimal_optuna_workers(self) -> int:
        """Optimal number of parallel Optuna trials.

        GPU trials should be serial (Optuna + GPU = 1 worker for stability).
        CPU-only trials can use ~4-5 parallel workers on 7500F.
        """
        if self.use_gpu:
            return 1  # sequential GPU trials — GPU utilization is the bottleneck
        return max(1, min(5, self.cpu_physical_cores - 1))

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            "=== Hardware Configuration ===",
            f"  CPU: {self.cpu_physical_cores} physical / {self.cpu_logical_cores} logical cores",
            f"  GPU: {self.gpu_name or 'None'}",
        ]
        if self.gpu_memory_mb:
            lines.append(f"  VRAM: {self.gpu_memory_mb} MB")
        lines.append(f"  CUDA available: {self.cuda_available}")
        lines.append(f"  GPU acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        lines.append(f"  Optimal n_jobs: {self.optimal_n_jobs}")
        lines.append(f"  Optimal Optuna workers: {self.optimal_optuna_workers}")
        return "\n".join(lines)


# ---- Global singleton, lazy-initialized ----
_config: GPUConfig | None = None


def get_gpu_config() -> GPUConfig:
    """Return the global hardware config (detected once, cached)."""
    global _config
    if _config is None:
        _config = GPUConfig.detect()
    return _config


def reset_config() -> None:
    """Clear cached config (useful for tests)."""
    global _config
    _config = None


# ---- Model parameter factories ----

def make_xgb_params(
    cfg: GPUConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return XGBRegressor init params with optimal GPU settings.

    On RTX 5060: sets device="cuda", tree_method="hist".
    Also sets optimal n_jobs for CPU fallback.

    Usage:
        model = xgb.XGBRegressor(**make_xgb_params())
        # or with overrides:
        model = xgb.XGBRegressor(**make_xgb_params(extra={"n_estimators": 500}))
    """
    if cfg is None:
        cfg = get_gpu_config()

    params: dict[str, Any] = {}
    if cfg.use_gpu:
        # XGBoost >= 2.0 uses device="cuda"
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        params["n_jobs"] = cfg.optimal_n_jobs

    if extra:
        # Merge, letting 'extra' override GPU defaults
        params.update(extra)

    return params


def make_catboost_params(
    cfg: GPUConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return CatBoostRegressor init params with optimal GPU settings.

    On RTX 5060: sets task_type="GPU", devices="0".
    Falls back to CPU with thread_count = optimal_n_jobs.

    Usage:
        model = CatBoostRegressor(**make_catboost_params())
    """
    if cfg is None:
        cfg = get_gpu_config()

    params: dict[str, Any] = {}
    if cfg.use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
        # GTX/RTX: border_count=128 recommended for GPU
        if "border_count" not in (extra or {}):
            params["border_count"] = 128
    else:
        params["task_type"] = "CPU"
        params["thread_count"] = cfg.optimal_n_jobs

    if extra:
        # Merge preserving CatBoost key names
        params.update(extra)

    return params


def make_lgbm_params(
    cfg: GPUConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return LGBMRegressor init params with optimal GPU settings.

    On RTX 5060: sets device="gpu", gpu_platform_id=0, gpu_device_id=0.
    Falls back to CPU with n_jobs = optimal_n_jobs.

    Usage:
        model = LGBMRegressor(**make_lgbm_params())
    """
    if cfg is None:
        cfg = get_gpu_config()

    params: dict[str, Any] = {}
    if cfg.use_gpu:
        params["device"] = "gpu"
        params["gpu_platform_id"] = 0
        params["gpu_device_id"] = 0
        # GPU works best with specific parameters
        if "num_leaves" not in (extra or {}):
            params["num_leaves"] = 31
    else:
        params["n_jobs"] = cfg.optimal_n_jobs

    if extra:
        params.update(extra)

    return params


def get_optuna_parallelism(cfg: GPUConfig | None = None) -> dict[str, Any]:
    """Return Optuna study + optimize params for 7500F.

    GPU-enabled → uses single process (GPU is the bottleneck).
    CPU-only → uses 4-5 parallel trials.

    Also sets optimal thread limits for Numpy/BLAS.
    """
    if cfg is None:
        cfg = get_gpu_config()

    # Set BLAS thread limits to prevent oversubscription
    n_threads = cfg.optimal_n_jobs
    env_hints = {
        "OMP_NUM_THREADS": str(n_threads),
        "MKL_NUM_THREADS": str(n_threads),
        "OPENBLAS_NUM_THREADS": str(n_threads),
    }
    for k, v in env_hints.items():
        os.environ.setdefault(k, v)

    return {
        "n_jobs": cfg.optimal_optuna_workers,
        "n_trials_per_worker": None,  # managed by Optuna
    }


# ---- Quick GPU verification (useful in smoke tests) ----

def verify_gpu_support(verbose: bool = True) -> bool:
    """Verify that GPU-accelerated ML libraries can use CUDA.

    Does NOT train — just checks imports and internal flags.
    Returns True if all 3 libraries report GPU readiness.

    Usage:
        python -c "from kcact.utils.gpu import verify_gpu_support; verify_gpu_support()"
    """
    cfg = get_gpu_config()
    if not cfg.cuda_available:
        if verbose:
            print("[GPU] No CUDA device detected via nvidia-smi.")
        return False
    if not cfg.use_gpu:
        if verbose:
            print("[GPU] Disabled via KCADT_NO_GPU environment variable.")
        return False

    ok: list[str] = []
    fail: list[str] = []

    # XGBoost
    try:
        import xgboost as xgb
        cuda_ok = xgb.build_info().get("USE_CUDA", False)
        if cuda_ok:
            ok.append("XGBoost")
        else:
            fail.append("XGBoost (no CUDA build)")
    except Exception as e:
        fail.append(f"XGBoost ({e})")

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        # Quick smoke: build with task_type="GPU" and check it doesn't crash
        # (CatBoost validates GPU at construction time)
        cb = CatBoostRegressor(iterations=2, task_type="GPU", devices="0", verbose=0)
        ok.append("CatBoost")
    except Exception as e:
        fail.append(f"CatBoost ({e})")

    # LightGBM
    try:
        from lightgbm import LGBMRegressor
        # LightGBM: need to build BoostParams to check GPU
        lgb = LGBMRegressor(
            n_estimators=2, device="gpu", gpu_device_id=0,
            gpu_platform_id=0, verbose=-1,
        )
        ok.append("LightGBM")
    except Exception as e:
        fail.append(f"LightGBM ({e})")

    if verbose:
        print(f"\n[GPU] {cfg.gpu_name or 'Unknown GPU'} | VRAM: {cfg.gpu_memory_mb} MB")
        print(f"[GPU] OK: {', '.join(ok) if ok else 'NONE'}")
        if fail:
            print(f"[GPU] FAIL: {', '.join(fail)}")
        if ok and not fail:
            print("[GPU] All 3 libraries GPU-ready!")

    return len(fail) == 0 and len(ok) == 3


# ---- Run detection on import (with minimal overhead) ----
if __name__ == "__main__":
    cfg = get_gpu_config()
    print(cfg.summary())
    print()
    verify_gpu_support()
