"""
backend/app/services/datasets/levir_cd.py
==========================================
LEVIR-CD dataset validator and PyTorch Dataset class.

LEVIR-CD is a large-scale remote sensing building change detection dataset.
Structure: <root>/{train,val,test}/{A,B,label}/
  A/     — "before" RGB images (PNG, 1024×1024)
  B/     — "after"  RGB images (PNG, 1024×1024)
  label/ — binary change masks (PNG, 1024×1024, L-mode, values 0/255)

The validator walks each split and checks:
  - A, B, label dirs all exist
  - Image counts match across A/B/label
  - Filenames match (by name, not path)
  - Sample images are readable and have expected format/dimensions
  - Detects corrupt files (unreadable by Pillow)
  - Reports missing pairs

The LEVIRDataset class is a torch.utils.data.Dataset.
  - Lazy loading: images are opened on __getitem__, not __init__
  - Does NOT load the whole dataset into RAM
  - Returns (img_A, img_B, label) as float32 tensors in [0, 1]
  - Label is normalised to {0.0, 1.0} from {0, 255}
  - Supports configurable crop size (random crop for train, centre crop for val/test)
  - Supports configurable split (train / val / test)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

from ...logging_setup import logger


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

@dataclass
class SplitValidation:
    split: str
    a_count: int = 0
    b_count: int = 0
    label_count: int = 0
    matched_triplets: int = 0
    mismatched_names: List[str] = field(default_factory=list)
    corrupt_files: List[str] = field(default_factory=list)
    missing_dirs: List[str] = field(default_factory=list)
    sample_checks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            len(self.missing_dirs) == 0
            and len(self.mismatched_names) == 0
            and len(self.corrupt_files) == 0
            and self.matched_triplets > 0
        )


@dataclass
class LEVIRValidationResult:
    root: str
    splits: Dict[str, SplitValidation] = field(default_factory=dict)
    global_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        if self.global_errors:
            return False
        return all(s.is_valid for s in self.splits.values())

    @property
    def total_triplets(self) -> int:
        return sum(s.matched_triplets for s in self.splits.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "valid": self.is_valid,
            "total_triplets": self.total_triplets,
            "splits": {
                name: {
                    "split": sv.split,
                    "a_count": sv.a_count,
                    "b_count": sv.b_count,
                    "label_count": sv.label_count,
                    "matched_triplets": sv.matched_triplets,
                    "is_valid": sv.is_valid,
                    "mismatched_names": sv.mismatched_names[:20],  # cap list size
                    "corrupt_files": sv.corrupt_files[:20],
                    "missing_dirs": sv.missing_dirs,
                    "sample_checks": sv.sample_checks,
                }
                for name, sv in self.splits.items()
            },
            "global_errors": self.global_errors,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_SAMPLE_CHECK_COUNT = 3  # Number of samples to inspect per split


def _list_images(directory: Path) -> List[Path]:
    """Return sorted list of image files in a directory (non-recursive)."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS
    )


def _check_sample(img_path: Path, expected_mode: Optional[str] = None) -> Dict[str, Any]:
    """Attempt to open an image and return basic metadata. Returns error on failure."""
    result: Dict[str, Any] = {"path": str(img_path), "ok": False}
    try:
        with Image.open(img_path) as img:
            result["ok"] = True
            result["size"] = list(img.size)  # [W, H]
            result["mode"] = img.mode
            if img.mode in ("L", "P"):
                arr = np.asarray(img)
                unique = np.unique(arr).tolist()
                result["unique_values"] = unique
                result["is_binary"] = set(unique).issubset({0, 255})
            if expected_mode and img.mode != expected_mode:
                result["mode_warning"] = f"Expected {expected_mode}, got {img.mode}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def validate_levir_cd(root: Path) -> LEVIRValidationResult:
    """
    Validate the LEVIR-CD dataset at ``root``.

    Checks all three splits (train, val, test). For each split it verifies:
    - A/, B/, and label/ subdirectories exist
    - Image counts match across A, B, label
    - Filenames match between A, B, label
    - A sample of files are readable by Pillow with expected format
    - Labels contain only {0, 255} values (binary)

    Does NOT load the entire dataset into RAM.
    Samples at most ``_SAMPLE_CHECK_COUNT`` images per sub-folder.

    Parameters
    ----------
    root : Path
        Root directory of the LEVIR-CD dataset.

    Returns
    -------
    LEVIRValidationResult
    """
    result = LEVIRValidationResult(root=str(root))

    if not root.exists():
        result.global_errors.append(f"Root directory does not exist: {root}")
        return result
    if not root.is_dir():
        result.global_errors.append(f"Root path is not a directory: {root}")
        return result

    for split in ("train", "val", "test"):
        sv = SplitValidation(split=split)
        split_dir = root / split

        if not split_dir.exists():
            sv.missing_dirs.append(str(split_dir))
            result.splits[split] = sv
            continue

        # Check subdirectories
        dir_a = split_dir / "A"
        dir_b = split_dir / "B"
        dir_label = split_dir / "label"

        for subdir, label in [(dir_a, "A"), (dir_b, "B"), (dir_label, "label")]:
            if not subdir.exists():
                sv.missing_dirs.append(f"{split}/{label}")

        if sv.missing_dirs:
            result.splits[split] = sv
            continue

        # List images in each folder
        images_a = _list_images(dir_a)
        images_b = _list_images(dir_b)
        images_l = _list_images(dir_label)

        sv.a_count = len(images_a)
        sv.b_count = len(images_b)
        sv.label_count = len(images_l)

        # Build name → path dicts for matching
        names_a = {f.name: f for f in images_a}
        names_b = {f.name: f for f in images_b}
        names_l = {f.name: f for f in images_l}

        all_names = set(names_a) | set(names_b) | set(names_l)
        matched = set(names_a) & set(names_b) & set(names_l)
        sv.matched_triplets = len(matched)

        # Report mismatches (names present in some but not all folders)
        for name in sorted(all_names - matched):
            in_a = name in names_a
            in_b = name in names_b
            in_l = name in names_l
            sv.mismatched_names.append(
                f"{name}: A={'✓' if in_a else '✗'} B={'✓' if in_b else '✗'} label={'✓' if in_l else '✗'}"
            )

        # Sample a few matched triplets for detailed checks
        sample_names = sorted(matched)[:_SAMPLE_CHECK_COUNT]
        for name in sample_names:
            triplet_info: Dict[str, Any] = {"name": name}

            # Check A (RGB expected)
            check_a = _check_sample(names_a[name], expected_mode="RGB")
            triplet_info["A"] = check_a
            if not check_a["ok"]:
                sv.corrupt_files.append(f"{split}/A/{name}")

            # Check B (RGB expected)
            check_b = _check_sample(names_b[name], expected_mode="RGB")
            triplet_info["B"] = check_b
            if not check_b["ok"]:
                sv.corrupt_files.append(f"{split}/B/{name}")

            # Check label (L-mode binary expected)
            check_l = _check_sample(names_l[name], expected_mode="L")
            triplet_info["label"] = check_l
            if not check_l["ok"]:
                sv.corrupt_files.append(f"{split}/label/{name}")
            elif not check_l.get("is_binary", False):
                triplet_info["label"]["warning"] = (
                    f"Expected binary mask with values {{0, 255}}, "
                    f"got {check_l.get('unique_values')}"
                )

            sv.sample_checks.append(triplet_info)

        result.splits[split] = sv
        logger.info(
            "[levir_cd] split=%s triplets=%d corrupt=%d mismatched=%d",
            split, sv.matched_triplets, len(sv.corrupt_files), len(sv.mismatched_names),
        )

    return result


# ---------------------------------------------------------------------------
# PyTorch Dataset (lazy loading, no full dataset in RAM)
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms.functional as TF
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Define a stub so the module is importable without torch
    class Dataset:  # type: ignore[no-redef]
        pass


class LEVIRDataset(Dataset):
    """
    PyTorch Dataset for LEVIR-CD binary change detection.

    Returns triplets (img_A, img_B, label) as float32 tensors.
    img_A, img_B : shape (3, H, W), values in [0, 1]
    label        : shape (1, H, W), values in {0.0, 1.0}

    Images are loaded lazily (one at a time in __getitem__) — the entire
    dataset is never in RAM simultaneously.

    Parameters
    ----------
    root : Path or str
        LEVIR-CD root directory.
    split : str
        One of "train", "val", "test".
    img_size : int
        Square size for cropping. Random crop for train, centre crop for val/test.
    augment : bool
        If True (train mode), apply random horizontal and vertical flips.
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: "Path | str",
        split: str = "train",
        img_size: int = 256,
        augment: bool = False,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchvision are required for LEVIRDataset. "
                "Install via: pip install torch torchvision"
            )
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

        root = Path(root)
        split_dir = root / split
        dir_a = split_dir / "A"
        dir_b = split_dir / "B"
        dir_label = split_dir / "label"

        for d, label in [(dir_a, "A"), (dir_b, "B"), (dir_label, "label")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"LEVIR-CD {split}/{label} directory not found: {d}"
                )

        # Build matched file lists
        images_a = {f.name: f for f in _list_images(dir_a)}
        images_b = {f.name: f for f in _list_images(dir_b)}
        images_l = {f.name: f for f in _list_images(dir_label)}
        matched_names = sorted(set(images_a) & set(images_b) & set(images_l))

        if not matched_names:
            raise ValueError(
                f"No matched A/B/label triplets found in {split_dir}. "
                "Ensure A/, B/, label/ contain images with matching filenames."
            )

        self.paths_a = [images_a[n] for n in matched_names]
        self.paths_b = [images_b[n] for n in matched_names]
        self.paths_l = [images_l[n] for n in matched_names]
        self.img_size = img_size
        self.augment = augment
        self.split = split

        logger.info(
            "[LEVIRDataset] split=%s img_size=%d triplets=%d augment=%s",
            split, img_size, len(self.paths_a), augment,
        )

    def __len__(self) -> int:
        return len(self.paths_a)

    def __getitem__(self, idx: int) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        img_a = Image.open(self.paths_a[idx]).convert("RGB")
        img_b = Image.open(self.paths_b[idx]).convert("RGB")
        label = Image.open(self.paths_l[idx]).convert("L")

        # --- Crop ---
        if self.split == "train" and self.augment:
            # Random crop
            i, j, h, w = self._get_random_crop_params(img_a, self.img_size)
        else:
            # Centre crop
            W, H = img_a.size
            i = (H - self.img_size) // 2
            j = (W - self.img_size) // 2
            h = self.img_size
            w = self.img_size

        img_a = TF.crop(img_a, i, j, h, w)
        img_b = TF.crop(img_b, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # --- Augmentation (train only) ---
        if self.augment and self.split == "train":
            if random.random() > 0.5:
                img_a = TF.hflip(img_a)
                img_b = TF.hflip(img_b)
                label = TF.hflip(label)
            if random.random() > 0.5:
                img_a = TF.vflip(img_a)
                img_b = TF.vflip(img_b)
                label = TF.vflip(label)

        # --- To tensor ---
        # Images: [0, 255] uint8 → [0.0, 1.0] float32, shape (3, H, W)
        t_a = TF.to_tensor(img_a)  # float32 in [0,1]
        t_b = TF.to_tensor(img_b)
        # Label: [0, 255] L-mode → {0.0, 1.0} float32, shape (1, H, W)
        t_l = TF.to_tensor(label)  # float32 in [0, 1] since 255→1.0
        # Binarise strictly (threshold at 0.5 in case of any anti-aliasing artefacts)
        t_l = (t_l > 0.5).float()

        return t_a, t_b, t_l

    @staticmethod
    def _get_random_crop_params(img: Image.Image, crop_size: int) -> Tuple[int, int, int, int]:
        """Return (i, j, h, w) for a random square crop of size crop_size."""
        W, H = img.size
        if H < crop_size or W < crop_size:
            raise ValueError(
                f"Image size ({W}×{H}) is smaller than requested crop_size ({crop_size}). "
                "Reduce img_size or use larger images."
            )
        i = random.randint(0, H - crop_size)
        j = random.randint(0, W - crop_size)
        return i, j, crop_size, crop_size
