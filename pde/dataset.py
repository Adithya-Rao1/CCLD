from __future__ import annotations

import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MULTIPHYSICS_PROBLEMS = ["TE_heat", "NS_heat", "E_flow", "MHD", "VA", "Elder"]
ALL_PROBLEMS = MULTIPHYSICS_PROBLEMS + ["diffusion_reaction"]

PROBLEM_SPECS = {
    "TE_heat": {
        "full_name": "Electro-Thermal",
        "coupling": "bidirectional",
        "input": [{"candidates": ["mater"], "ext": "mat"}],
        "outputs": [
            {"candidates": ["Ez"], "label": "Ez", "kind": "complex", "ext": "mat"},
            {"candidates": ["T", "u_T"], "label": "T", "kind": "real", "ext": "mat"},
        ],
    },
    "NS_heat": {
        "full_name": "Thermo-Fluid",
        "coupling": "bidirectional",
        "input": [{"candidates": ["Q_heat"], "ext": "csv"}],
        "outputs": [
            {"candidates": ["u_u"], "label": "u", "kind": "real", "ext": "csv"},
            {"candidates": ["u_v"], "label": "v", "kind": "real", "ext": "csv"},
            {"candidates": ["u_T", "T"], "label": "T", "kind": "real", "ext": "csv"},
        ],
    },
    "E_flow": {
        "full_name": "Electro-Fluid",
        "coupling": "unidirectional",
        "input": [{"candidates": ["kappa"], "ext": "mat"}],
        "outputs": [
            {"candidates": ["ec_V"], "label": "ec_V", "kind": "real", "ext": "mat"},
            {"candidates": ["u_flow"], "label": "u_flow", "kind": "real", "ext": "mat"},
            {"candidates": ["v_flow"], "label": "v_flow", "kind": "real", "ext": "mat"},
        ],
    },
    "MHD": {
        "full_name": "Magneto-Hydrodynamic",
        "coupling": "bidirectional",
        "input": [{"candidates": ["Br"], "ext": "mat"}],
        "outputs": [
            {"candidates": ["Jx"], "label": "Jx", "kind": "real", "ext": "mat"},
            {"candidates": ["Jy"], "label": "Jy", "kind": "real", "ext": "mat"},
            {"candidates": ["Jz"], "label": "Jz", "kind": "real", "ext": "mat"},
            {"candidates": ["u_u"], "label": "u_u", "kind": "real", "ext": "mat"},
            {"candidates": ["u_v"], "label": "u_v", "kind": "real", "ext": "mat"},
        ],
    },
    "VA": {
        "full_name": "Acoustic-Structure",
        "coupling": "bidirectional",
        "input": [{"candidates": ["rho_water"], "ext": "mat"}],
        "outputs": [
            {"candidates": ["p_t"], "label": "p_t", "kind": "complex", "ext": "mat"},
            {"candidates": ["Sxx"], "label": "Sxx", "kind": "complex", "ext": "mat"},
            {"candidates": ["Sxy"], "label": "Sxy", "kind": "complex", "ext": "mat"},
            {"candidates": ["Syy"], "label": "Syy", "kind": "complex", "ext": "mat"},
            {"candidates": ["x_u"], "label": "x_u", "kind": "complex", "ext": "mat"},
            {"candidates": ["x_v"], "label": "x_v", "kind": "complex", "ext": "mat"},
        ],
    },
    "Elder": {
        "full_name": "Mass-Transport-Fluid",
        "coupling": "bidirectional",
    },
}

DIFFUSION_REACTION_METADATA = {
    "full_name": "Diffusion-Reaction (PDEBench)",
    "coupling": "bidirectional", 
}

ELDER_ROLLOUT_FIELDS = ["u_u", "u_v", "c_flow"]
ELDER_N_TIMESTEPS = 10
ELDER_TIMESTEP_OFFSET = 1  

TASK_SUBSETS = {
    ("TE_heat", 2): ["Re{Ez}", "T"],
    ("diffusion_reaction", 2): ["v1", "v2"],
}


def _load_mat_field(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]
    if not keys:
        raise ValueError
    if len(keys) == 1:
        return np.asarray(mat[keys[0]])
    stem = os.path.splitext(os.path.basename(path))[0]
    for k in keys:
        if k.lower() == stem.lower():
            return np.asarray(mat[k])
    return np.asarray(mat[keys[0]])


def _load_csv_field(path: str) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=np.float64)


def _to_chw_tensor(arr: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        raise ValueError
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        n = arr.shape[0]
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError
        arr = arr.reshape(side, side)
    if arr.ndim != 2:
        raise ValueError
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0)
    if tuple(t.shape[-2:]) != tuple(size):
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t.squeeze(0)


def _complex_to_real_imag(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr)
    return np.real(arr), np.imag(arr)


def _find_field_dir(problem_root: str, candidates: List[str]) -> str:
    for c in candidates:
        direct = os.path.join(problem_root, c)
        if os.path.isdir(direct):
            return direct
    if os.path.isdir(problem_root):
        present = {name.lower(): name for name in os.listdir(problem_root)}
        for c in candidates:
            if c.lower() in present:
                return os.path.join(problem_root, present[c.lower()])
    available = sorted(os.listdir(problem_root)) if os.path.isdir(problem_root) else []
    raise FileNotFoundError(
        f"None of the candidate field directories {candidates} exist under {problem_root} "
        f"(this candidate name is a best-effort guess -- see PROBLEM_SPECS docstring). "
        f"Directories actually present: {available}. Edit PROBLEM_SPECS in dataset.py to match."
    )


def _list_sample_indices(field_dir: str, is_dir_per_sample: bool = False) -> List[int]:
    entries = os.listdir(field_dir)
    idxs = []
    for e in entries:
        stem = e if is_dir_per_sample else os.path.splitext(e)[0]
        if re.fullmatch(r"\d+", stem):
            idxs.append(int(stem))
    if not idxs:
        raise FileNotFoundError(f"No numerically-named samples found under {field_dir}")
    return sorted(idxs)


def _standard_split_sample_indices(problem_root: str, spec: Dict) -> List[int]:
    try:
        input_dirs = [_find_field_dir(problem_root, f["candidates"]) for f in spec["input"]]
        output_dirs = [_find_field_dir(problem_root, f["candidates"]) for f in spec["outputs"]]
    except FileNotFoundError:
        return []
    try:
        idxs = set(_list_sample_indices(output_dirs[0]))
        for d in input_dirs + output_dirs[1:]:
            idxs &= set(_list_sample_indices(d))
    except FileNotFoundError:
        return []
    return sorted(idxs)


def _testing_split_broken(root_dir: str, problem: str) -> bool:
    testing_root = os.path.join(root_dir, "testing", problem)
    if not os.path.isdir(testing_root):
        return False
    spec = PROBLEM_SPECS[problem]
    return not _standard_split_sample_indices(testing_root, spec)


_HELD_OUT_SPLIT_DIR = os.path.join(os.path.dirname(__file__), "held_out_splits")


def _held_out_split_path(problem: str) -> str:
    return os.path.join(_HELD_OUT_SPLIT_DIR, f"{problem}_held_out_split.pt")


def _get_or_create_held_out_split(problem_root: str, spec: Dict, problem: str) -> Dict:
    path = _held_out_split_path(problem)
    if os.path.isfile(path):
        cached = torch.load(path)
        if cached.get("source_problem_root") == problem_root:
            return cached
    idxs = _standard_split_sample_indices(problem_root, spec)
    if not idxs:
        raise FileNotFoundError(f"No usable samples found under {problem_root} to build a held-out split from")
    n_total = len(idxs)
    n_train = int(round(n_total * 0.9))
    if n_total >= 2:
        n_train = min(max(n_train, 1), n_total - 1)
    else:
        n_train = n_total
    split = {"train": idxs[:n_train], "test": idxs[n_train:], "source_problem_root": problem_root, "n_total": n_total}
    if not os.path.isfile(path):
        os.makedirs(_HELD_OUT_SPLIT_DIR, exist_ok=True)
        torch.save(split, path)
    return split


_TARGET_NORM_STATS_DIR = os.path.join(os.path.dirname(__file__), "target_norm_stats")


def _target_norm_stats_path(problem: str) -> str:
    return os.path.join(_TARGET_NORM_STATS_DIR, f"{problem}_target_norm_stats.pt")


def compute_target_norm_stats(dataset, task_names: List[str]) -> Dict[str, Dict[str, float]]:
    sums = {name: 0.0 for name in task_names}
    sumsqs = {name: 0.0 for name in task_names}
    counts = {name: 0 for name in task_names}
    for i in range(len(dataset)):
        sample = dataset[i]
        for name, field in zip(sample["task_names"], sample["tasks"]):
            arr = field.double()
            sums[name] += arr.sum().item()
            sumsqs[name] += (arr ** 2).sum().item()
            counts[name] += arr.numel()
    stats = {}
    for name in task_names:
        mean = sums[name] / counts[name]
        var = max(sumsqs[name] / counts[name] - mean ** 2, 1e-12)
        stats[name] = {"mean": mean, "std": max(var ** 0.5, 1e-6)}
    return stats


def _get_or_create_target_norm_stats(dataset, problem: str, task_names: List[str]) -> Dict[str, Dict[str, float]]:
    path = _target_norm_stats_path(problem)
    key = {"task_names": sorted(task_names), "n_samples": len(dataset)}
    if os.path.isfile(path):
        cached = torch.load(path)
        if cached.get("_key") == key:
            return cached["stats"]
    stats = compute_target_norm_stats(dataset, task_names)
    if not os.path.isfile(path):
        os.makedirs(_TARGET_NORM_STATS_DIR, exist_ok=True)
        torch.save({"_key": key, "stats": stats}, path)
    return stats


def _find_h5_file(root_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(root_dir, "**", "*diff-react*.h5"), recursive=True))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True))
    if not candidates:
        raise FileNotFoundError(
            f"No PDEBench .h5 file found under {root_dir}; run download_pdebench.py first."
        )
    return candidates[0]


def _load_pdebench_diffusion_reaction(root_dir: str) -> np.ndarray:
    path = _find_h5_file(root_dir)
    with h5py.File(path, "r") as f:
        if "data" in f and isinstance(f["data"], h5py.Dataset):
            arr = np.asarray(f["data"])
        else:
            keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))
            if not keys:
                raise ValueError
            arrs = []
            for k in keys:
                grp = f[k]
                if "data" not in grp:
                    raise ValueError
                arrs.append(np.asarray(grp["data"]))
            arr = np.stack(arrs, axis=0)
    if arr.ndim != 5 or arr.shape[-1] < 2:
        raise ValueError
    return arr


class MultiPhysicsFieldDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        problem: str,
        split: str = "training",
        n_tasks: Optional[int] = None,
        task_subset: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (128, 128),
        max_samples: Optional[int] = None,
        pde_snapshot: Optional[int] = None,
        pde_second_snapshot: Optional[int] = None,
    ):
        if problem not in ALL_PROBLEMS:
            raise ValueError
        self.root_dir = root_dir
        self.problem = problem
        self.split = split
        self.image_size = tuple(image_size)
        self.max_samples = max_samples

        if problem == "diffusion_reaction":
            self._mode = "diffusion_reaction"
            self._init_diffusion_reaction(n_tasks, task_subset, pde_snapshot, pde_second_snapshot)
        elif problem == "Elder":
            self._mode = "elder"
            self._init_elder()
        else:
            self._mode = "standard"
            self._init_standard(n_tasks, task_subset)

    def _init_diffusion_reaction(self, n_tasks, task_subset, pde_snapshot, pde_second_snapshot):
        self._pde_array = _load_pdebench_diffusion_reaction(self.root_dir)  # (N,T,X,Y,C)
        n_samples_total, n_t = self._pde_array.shape[0], self._pde_array.shape[1]

        split_key = str(self.split).lower()
        train_keys = {"training", "train"}
        test_keys = {"testing", "test", "val", "validation"}
        if split_key not in train_keys and split_key not in test_keys:
            raise ValueError
        n_train = int(round(n_samples_total * 0.9))
        if n_samples_total >= 2:
            n_train = min(max(n_train, 1), n_samples_total - 1)
        else:
            n_train = n_samples_total
        if split_key in train_keys:
            split_indices = list(range(0, n_train))
        else:
            split_indices = list(range(n_train, n_samples_total))

        if self.max_samples is not None:
            split_indices = split_indices[: self.max_samples]
        self._sample_indices = split_indices

        native_labels = ["f1", "f2", "u1", "u2"]
        if task_subset is not None:
            labels = list(task_subset)
        elif n_tasks == 2:
            labels = TASK_SUBSETS[("diffusion_reaction", 2)]
        elif n_tasks is None or n_tasks == 4:
            labels = native_labels
        else:
            raise ValueError
        self.task_names = labels

        snap1 = pde_snapshot if pde_snapshot is not None else n_t // 2
        snap2 = pde_second_snapshot if pde_second_snapshot is not None else n_t - 1
        self._pde_snapshots = {"v1": (0, snap1), "v2": (1, snap1), "f1": (0, snap1), "f2": (1, snap1),
                                "u1": (0, snap2), "u2": (1, snap2)}

    def _init_elder(self):
        problem_root = os.path.join(self.root_dir, self.split, "Elder")
        self._elder_dirs = {f: _find_field_dir(problem_root, [f]) for f in ["S_c"] + ELDER_ROLLOUT_FIELDS}
        idxs = _list_sample_indices(self._elder_dirs[ELDER_ROLLOUT_FIELDS[0]], is_dir_per_sample=True)
        if self.max_samples is not None:
            idxs = idxs[: self.max_samples]
        self._sample_indices = idxs
        self.task_names = [f"{f}_t{k}" for k in range(1, ELDER_N_TIMESTEPS + 1) for f in ELDER_ROLLOUT_FIELDS]
        self.rollout_shape = (ELDER_N_TIMESTEPS, len(ELDER_ROLLOUT_FIELDS))

    def _init_standard(self, n_tasks, task_subset):
        spec = PROBLEM_SPECS[self.problem]
        split_key = str(self.split).lower()
        train_keys = {"training", "train"}
        test_keys = {"testing", "test", "val", "validation"}
        self._use_held_out_split = (
            split_key in (train_keys | test_keys) and _testing_split_broken(self.root_dir, self.problem)
        )
        if self._use_held_out_split:
            problem_root = os.path.join(self.root_dir, "training", self.problem)
        else:
            problem_root = os.path.join(self.root_dir, self.split, self.problem)
        self._input_dirs = [(_find_field_dir(problem_root, f["candidates"]), f["ext"]) for f in spec["input"]]

        resolved_outputs = []
        for f in spec["outputs"]:
            d = _find_field_dir(problem_root, f["candidates"])
            resolved_outputs.append({**f, "dir": d})
        self._output_specs = resolved_outputs

        native_labels = []
        for f in resolved_outputs:
            if f["kind"] == "complex":
                native_labels += [f"Re{{{f['label']}}}", f"Im{{{f['label']}}}"]
            else:
                native_labels.append(f["label"])

        if task_subset is not None:
            labels = list(task_subset)
        elif n_tasks is not None and n_tasks != len(native_labels):
            key = (self.problem, n_tasks)
            if key not in TASK_SUBSETS:
                raise ValueError
            labels = TASK_SUBSETS[key]
        else:
            labels = native_labels
        unknown = [l for l in labels if l not in native_labels]
        if unknown:
            raise ValueError
        self.task_names = labels

        if self._use_held_out_split:
            held_out = _get_or_create_held_out_split(problem_root, spec, self.problem)
            idxs = held_out["train"] if split_key in train_keys else held_out["test"]
        else:
            idxs = _standard_split_sample_indices(problem_root, spec)
            if not idxs:
                checked = [d for d, _ in self._input_dirs] + [f["dir"] for f in resolved_outputs]
                raise FileNotFoundError(
                    f"No sample indices are present in ALL of {self.problem}'s field directories "
                    f"under {problem_root} ({self.split} split). Checked: {checked}"
                )
        if self.max_samples is not None:
            idxs = idxs[: self.max_samples]
        self._sample_indices = idxs
        self._elliptic_dir = os.path.join(problem_root, "ellipticcsv") if self.problem == "TE_heat" else None

    def __len__(self) -> int:
        return len(self._sample_indices)

    def _load_field_file(self, dir_path: str, sample_idx: int, ext: str) -> np.ndarray:
        path = os.path.join(dir_path, f"{sample_idx}.{ext}")
        return _load_mat_field(path) if ext == "mat" else _load_csv_field(path)

    def _get_standard(self, i: int) -> Dict:
        idx = self._sample_indices[i]
        cond_channels = [
            _to_chw_tensor(self._load_field_file(d, idx, ext), self.image_size)
            for d, ext in self._input_dirs
        ]
        conditioning = torch.cat(cond_channels, dim=0)

        by_label: Dict[str, torch.Tensor] = {}
        for f in self._output_specs:
            produced = [f"Re{{{f['label']}}}", f"Im{{{f['label']}}}"] if f["kind"] == "complex" else [f["label"]]
            if not any(p in self.task_names for p in produced):
                continue
            arr = self._load_field_file(f["dir"], idx, f["ext"])
            if f["kind"] == "complex":
                re, im = _complex_to_real_imag(arr)
                if produced[0] in self.task_names:
                    by_label[produced[0]] = _to_chw_tensor(re, self.image_size)
                if produced[1] in self.task_names:
                    by_label[produced[1]] = _to_chw_tensor(im, self.image_size)
            else:
                by_label[f["label"]] = _to_chw_tensor(arr, self.image_size)

        tasks = [by_label[name] for name in self.task_names]
        sample = {"conditioning": conditioning, "tasks": tasks, "task_names": self.task_names}
        if self._elliptic_dir is not None:
            arr = _load_csv_field(os.path.join(self._elliptic_dir, f"{idx}.csv"))
            sample["elliptic_params"] = torch.as_tensor(arr, dtype=torch.float32).reshape(-1)[:3]
        return sample

    def _get_elder(self, i: int) -> Dict:
        idx = self._sample_indices[i]
        s_c = _to_chw_tensor(_load_mat_field(os.path.join(self._elder_dirs["S_c"], f"{idx}.mat")), self.image_size)
        t0_channels = [s_c]
        for f in ELDER_ROLLOUT_FIELDS:
            t0_channels.append(
                _to_chw_tensor(_load_mat_field(os.path.join(self._elder_dirs[f], str(idx), "0.mat")), self.image_size)
            )
        conditioning = torch.cat(t0_channels, dim=0)

        tasks = []
        for k in range(ELDER_TIMESTEP_OFFSET, ELDER_TIMESTEP_OFFSET + ELDER_N_TIMESTEPS):
            for f in ELDER_ROLLOUT_FIELDS:
                arr = _load_mat_field(os.path.join(self._elder_dirs[f], str(idx), f"{k}.mat"))
                tasks.append(_to_chw_tensor(arr, self.image_size))
        return {"conditioning": conditioning, "tasks": tasks, "task_names": self.task_names}

    def _get_diffusion_reaction(self, i: int) -> Dict:
        idx = self._sample_indices[i]
        sample = self._pde_array[idx]  # (T,X,Y,C)
        t0 = sample[0]
        conditioning = torch.cat(
            [_to_chw_tensor(t0[..., 0], self.image_size), _to_chw_tensor(t0[..., 1], self.image_size)],
            dim=0,
        )
        tasks = []
        for name in self.task_names:
            ch, snap = self._pde_snapshots[name]
            tasks.append(_to_chw_tensor(sample[snap, ..., ch], self.image_size))
        return {"conditioning": conditioning, "tasks": tasks, "task_names": self.task_names}

    def __getitem__(self, i: int) -> Dict:
        if self._mode == "standard":
            return self._get_standard(i)
        if self._mode == "elder":
            return self._get_elder(i)
        return self._get_diffusion_reaction(i)


def collate_fn(batch: List[Dict]) -> Dict:
    conditioning = torch.stack([b["conditioning"] for b in batch], dim=0)
    n_tasks = len(batch[0]["tasks"])
    tasks = [torch.stack([b["tasks"][i] for b in batch], dim=0) for i in range(n_tasks)]
    out = {"conditioning": conditioning, "tasks": tasks, "task_names": batch[0]["task_names"]}
    if "elliptic_params" in batch[0]:
        out["elliptic_params"] = torch.stack([b["elliptic_params"] for b in batch], dim=0)
    return out