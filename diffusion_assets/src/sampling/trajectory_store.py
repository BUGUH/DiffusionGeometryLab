"""轨迹持久化存储，支持内存和 HDF5 后端。"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """
    存储多次采样的轨迹数据，支持按 (run_id, t) 索引。

    后端:
    - "memory": 纯内存字典
    - "hdf5": HDF5 文件
    """

    def __init__(self, backend: str = "memory", save_dir: Optional[str] = None):
        self.backend = backend
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/trajectories")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, Dict[float, np.ndarray]] = {}

    def save_trajectory(self, run_id: str, t_val: float, data: torch.Tensor):
        """存储单个时间步的数据。"""
        arr = data.detach().cpu().numpy()

        if self.backend == "memory":
            if run_id not in self._memory:
                self._memory[run_id] = {}
            self._memory[run_id][t_val] = arr

        elif self.backend == "hdf5":
            self._save_hdf5(run_id, t_val, arr)

    def load_trajectory(self, run_id: str, t_val: float) -> Optional[torch.Tensor]:
        """读取单个时间步的数据。"""
        if self.backend == "memory":
            arr = self._memory.get(run_id, {}).get(t_val)
            return torch.from_numpy(arr) if arr is not None else None

        elif self.backend == "hdf5":
            return self._load_hdf5(run_id, t_val)

    def save_bundle(self, run_id: str, bundle):
        """保存完整 TrajectoryBundle。"""
        for t_val, x_t in bundle.x_ts.items():
            self.save_trajectory(f"{run_id}/x_t", t_val, x_t)
        for t_val, score in bundle.scores.items():
            self.save_trajectory(f"{run_id}/score", t_val, score)
        for t_val, pred_x0 in bundle.pred_x0s.items():
            self.save_trajectory(f"{run_id}/pred_x0", t_val, pred_x0)
        logger.info(f"轨迹已保存: run={run_id}, {len(bundle.x_ts)} 个时间步")

    def list_runs(self):
        if self.backend == "memory":
            return list(self._memory.keys())
        return [f.stem for f in self.save_dir.glob("*.h5")]

    def _save_hdf5(self, run_id: str, t_val: float, arr: np.ndarray):
        try:
            import h5py
        except ImportError:
            raise ImportError("HDF5 后端需要 h5py: pip install h5py")

        safe_id = run_id.replace("/", "_")
        path = self.save_dir / f"{safe_id}.h5"
        with h5py.File(path, "a") as f:
            key = f"t_{t_val:.6f}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=arr, compression="gzip")

    def _load_hdf5(self, run_id: str, t_val: float) -> Optional[torch.Tensor]:
        try:
            import h5py
        except ImportError:
            return None

        safe_id = run_id.replace("/", "_")
        path = self.save_dir / f"{safe_id}.h5"
        if not path.exists():
            return None
        with h5py.File(path, "r") as f:
            key = f"t_{t_val:.6f}"
            if key in f:
                return torch.from_numpy(f[key][:])
        return None