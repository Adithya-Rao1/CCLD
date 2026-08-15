from __future__ import annotations

import io
import os
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from image import pseudo_labels as pl

ALL_TASKS = ["segmentation", "depth", "normals", "edges", "saliency", "human_parts"]

# human_parts would need PASCAL-Person-Part annotations, not present in either source, so it stays
# unsupported everywhere for now.
SOURCE_TASKS = {
    "nyudv2": {"depth", "normals", "edges", "saliency", "segmentation"},
    "pascal": {"segmentation", "edges", "saliency"},
    "cityscapes": {"segmentation", "edges", "saliency"},
}

NUM_CLASSES = {"pascal": 21, "cityscapes": 19}  # nyudv2 is inferred at construction time
SEG_IGNORE_INDEX = 255

# cityscapesScripts labels.py labelId -> trainId table (unlisted ids map to ignore=255)
_CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18,
}
_CITYSCAPES_LUT = np.full(256, SEG_IGNORE_INDEX, dtype=np.int64)
for _labelid, _trainid in _CITYSCAPES_ID_TO_TRAINID.items():
    _CITYSCAPES_LUT[_labelid] = _trainid

_NYUDV2_SPLIT_PREFIX = {"train": "train", "val": "validation"}


def task_output_channels(task: str, num_classes: Optional[int] = None) -> int:
    if task == "segmentation":
        if num_classes is None:
            raise ValueError("segmentation requires num_classes")
        return num_classes
    return {"depth": 1, "normals": 3, "edges": 1, "saliency": 1}[task]


def _find_dir(root: str, required_subdirs: List[str], max_depth: int = 4) -> Optional[str]:
    root = os.path.abspath(root)
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
        if all(os.path.isdir(os.path.join(dirpath, s)) for s in required_subdirs):
            return dirpath
    return None


def _index_pascal(root: str, split: str) -> List[Dict]:
    voc_root = _find_dir(root, ["JPEGImages", "SegmentationClass"])
    if voc_root is None:
        raise FileNotFoundError(
            f"Could not find a VOCdevkit/VOC2012 layout (JPEGImages/ + SegmentationClass/) under "
            f"{root}. Re-run download_pascal.py, or point --data-root at the extracted VOCdevkit "
            "directory."
        )
    split_name = "train" if split == "train" else "val"
    split_file = os.path.join(voc_root, "ImageSets", "Segmentation", f"{split_name}.txt")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(
            f"Missing PASCAL split file {split_file}. Expected the standard VOC devkit layout "
            "produced by extracting VOCtrainval_11-May-2012.zip via download_pascal.py."
        )
    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    items = []
    for image_id in ids:
        image_path = os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg")
        seg_path = os.path.join(voc_root, "SegmentationClass", f"{image_id}.png")
        if os.path.isfile(image_path) and os.path.isfile(seg_path):
            items.append({"image": image_path, "segmentation": seg_path})
    if not items:
        raise FileNotFoundError(f"Split file {split_file} matched no image/mask pairs under {voc_root}.")
    return items


def _index_cityscapes(root: str, split: str) -> List[Dict]:
    img_root = _find_dir(root, ["leftImg8bit"])
    gt_root = _find_dir(root, ["gtFine"])
    if img_root is None or gt_root is None:
        raise FileNotFoundError(
            f"Could not find leftImg8bit/ and gtFine/ directories under {root}. Re-run "
            "download_cityscapes.py with --packages gtFine_trainvaltest,leftImg8bit_trainvaltest."
        )
    split_name = "train" if split == "train" else "val"
    img_split_dir = os.path.join(img_root, "leftImg8bit", split_name)
    gt_split_dir = os.path.join(gt_root, "gtFine", split_name)
    if not os.path.isdir(img_split_dir) or not os.path.isdir(gt_split_dir):
        raise FileNotFoundError(
            f"Missing split directories {img_split_dir} / {gt_split_dir}. Expected the official "
            "cityscapesScripts layout: leftImg8bit/<split>/<city>/*.png, gtFine/<split>/<city>/*.png."
        )
    items = []
    for city in sorted(os.listdir(img_split_dir)):
        city_img_dir = os.path.join(img_split_dir, city)
        city_gt_dir = os.path.join(gt_split_dir, city)
        if not os.path.isdir(city_img_dir):
            continue
        for fn in sorted(os.listdir(city_img_dir)):
            if not fn.endswith("_leftImg8bit.png"):
                continue
            prefix = fn[: -len("_leftImg8bit.png")]
            seg_path = os.path.join(city_gt_dir, f"{prefix}_gtFine_labelIds.png")
            if os.path.isfile(seg_path):
                items.append({"image": os.path.join(city_img_dir, fn), "segmentation": seg_path})
    if not items:
        raise FileNotFoundError(f"No matched image/label pairs found under {img_split_dir}.")
    return items


def _load_nyudv2_table(root: str, split: str) -> pa.Table:
    prefix = _NYUDV2_SPLIT_PREFIX.get(split, split)
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".parquet") and fn.split("-")[0] == prefix:
                matches.append(os.path.join(dirpath, fn))
    if not matches:
        raise FileNotFoundError(
            f"No '{prefix}-*.parquet' shards found under {root}. Re-run download_nyudv2.py, which "
            "fetches the refs/convert/parquet revision of jagennath-hari/nyuv2 (rgb/depth/semantic/"
            "instance columns stored as embedded-image parquet shards named '<split>-N-of-M.parquet')."
        )
    tables = [pq.read_table(m) for m in sorted(matches)]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def _decode_hf_image(cell) -> Image.Image:
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is not None:
            return Image.open(io.BytesIO(data))
        path = cell.get("path")
        if path:
            return Image.open(path)
        raise ValueError(f"HF image cell has neither bytes nor path: {cell!r}")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell))
    if isinstance(cell, str):
        return Image.open(cell)
    raise ValueError(f"Unrecognized HF image cell type: {type(cell)}")


def _infer_nyudv2_num_classes(table: pa.Table, sample: int = 32) -> int:
    """heuristic: max observed label (excluding 255-as-ignore) over a row sample, +1. Override via
    MultiTaskVisionDataset(..., num_classes=...) once the true scheme is known for a given split."""
    n = min(sample, table.num_rows)
    max_label = 0
    col = table.column("semantic")
    for i in range(n):
        arr = np.asarray(_decode_hf_image(col[i].as_py()))
        if arr.ndim == 3:
            arr = arr[..., 0]
        local_max = int(arr.max())
        if local_max < 255:
            max_label = max(max_label, local_max)
    return max_label + 1


def _rgb_image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _depth_image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    scale = 65535.0 if img.mode in ("I", "I;16", "I;16B", "I;16L") else 255.0
    return torch.from_numpy(arr / scale).unsqueeze(0)


def _seg_image_to_tensor(img: Image.Image, apply_cityscapes_lut: bool) -> torch.Tensor:
    arr = np.asarray(img).astype(np.int64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if apply_cityscapes_lut:
        arr = _CITYSCAPES_LUT[arr]
    return torch.from_numpy(arr).long()


class MultiTaskVisionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        task_list: List[str],
        source: str,
        split: str = "train",
        height: int = 128,
        width: int = 128,
        num_classes: Optional[int] = None,
    ):
        if source not in SOURCE_TASKS:
            raise ValueError(f"Unknown source {source!r}; choose from {list(SOURCE_TASKS)}")
        unsupported = set(task_list) - SOURCE_TASKS[source]
        if unsupported:
            raise ValueError(
                f"source={source!r} does not support task(s) {sorted(unsupported)}. "
                f"Supported for this source: {sorted(SOURCE_TASKS[source])}."
            )

        self.root_dir = root_dir
        self.task_list = list(task_list)
        self.source = source
        self.split = split
        self.height = height
        self.width = width
        self._table = None
        self.items = None

        if source == "nyudv2":
            self._table = _load_nyudv2_table(root_dir, split)
            self.num_classes = num_classes
            if self.num_classes is None and "segmentation" in self.task_list:
                self.num_classes = _infer_nyudv2_num_classes(self._table)
        else:
            self.items = _index_pascal(root_dir, split) if source == "pascal" else _index_cityscapes(root_dir, split)
            self.num_classes = num_classes if num_classes is not None else NUM_CLASSES.get(source)

    def __len__(self) -> int:
        return self._table.num_rows if self._table is not None else len(self.items)

    def _raw_sample(self, idx: int) -> Dict[str, Image.Image]:
        need_depth = "depth" in self.task_list or "normals" in self.task_list
        need_seg = "segmentation" in self.task_list

        if self._table is not None:
            raw = {"image": _decode_hf_image(self._table.column("rgb")[idx].as_py())}
            if need_depth:
                raw["depth"] = _decode_hf_image(self._table.column("depth")[idx].as_py())
            if need_seg:
                raw["segmentation"] = _decode_hf_image(self._table.column("semantic")[idx].as_py())
            return raw

        item = self.items[idx]
        raw = {"image": Image.open(item["image"]).convert("RGB")}
        if need_depth:
            raw["depth"] = Image.open(item["depth"])
        if need_seg:
            raw["segmentation"] = Image.open(item["segmentation"])
        return raw

    def _resize(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        x = x.unsqueeze(0)
        kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
        x = F.interpolate(x, size=(self.height, self.width), mode=mode, **kwargs)
        return x.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw = self._raw_sample(idx)
        image = self._resize(_rgb_image_to_tensor(raw["image"]), mode="bilinear")
        sample = {"image": image}

        depth_resized = None
        if "depth" in raw:
            depth_resized = self._resize(_depth_image_to_tensor(raw["depth"]), mode="bilinear")

        for task in self.task_list:
            if task == "segmentation":
                seg = _seg_image_to_tensor(raw["segmentation"], apply_cityscapes_lut=self.source == "cityscapes")
                sample["segmentation"] = self._resize(seg.unsqueeze(0).float(), mode="nearest").squeeze(0).long()
            elif task == "depth":
                sample["depth"] = depth_resized
            elif task == "normals":
                sample["normals"] = pl.depth_to_normals(depth_resized)
            elif task == "edges":
                sample["edges"] = pl.image_to_edges(image)
            elif task == "saliency":
                sample["saliency"] = pl.saliency_proxy(image)
            elif task == "human_parts":
                raise NotImplementedError(
                    "human_parts requires PASCAL-Person-Part annotations, not present in either "
                    "the Ultralytics VOC mirror or the jagennath-hari/nyuv2 HF dataset."
                )
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}