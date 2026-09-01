"""
tests/test_levir_cd.py
-----------------------
Tests for LEVIR-CD dataset validator and LEVIRDataset.

All tests use temporary directory fixtures — no actual LEVIR-CD dataset
download is required for the tests to pass. Image files are tiny synthetic
PNGs generated in-process.

Tests are CPU-safe and fast (< 5 seconds total).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.datasets.levir_cd import validate_levir_cd, LEVIRValidationResult


# ---------------------------------------------------------------------------
# Fixtures — build tiny valid LEVIR-CD directory structures in tmp_path
# ---------------------------------------------------------------------------

def _create_rgb_png(path: Path, size=(8, 8), color=(100, 150, 200)):
    """Write a tiny RGB PNG to path."""
    Image.new("RGB", size, color=color).save(path, format="PNG")


def _create_binary_label_png(path: Path, size=(8, 8)):
    """Write a tiny binary L-mode PNG (values 0 and 255) to path."""
    arr = np.zeros(size[::-1], dtype=np.uint8)
    arr[2:5, 2:5] = 255  # changed region
    Image.fromarray(arr, mode="L").save(path, format="PNG")


def _build_valid_levir_root(tmp_path: Path, names=("img_1.png", "img_2.png")) -> Path:
    """
    Create a minimal LEVIR-CD directory structure with matching triplets.
    Returns the root path.
    """
    root = tmp_path / "LEVIR-CD"
    for split in ("train", "val", "test"):
        for subdir in ("A", "B", "label"):
            d = root / split / subdir
            d.mkdir(parents=True)
            for name in names:
                if subdir in ("A", "B"):
                    _create_rgb_png(d / name)
                else:
                    _create_binary_label_png(d / name)
    return root


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_levir_cd_with_valid_fixture(tmp_path):
    """
    A properly structured LEVIR-CD directory with matching A/B/label files
    should pass validation.
    """
    root = _build_valid_levir_root(tmp_path)
    result = validate_levir_cd(root)

    assert isinstance(result, LEVIRValidationResult)
    assert result.is_valid, f"Expected valid, got global_errors={result.global_errors}, splits={result.splits}"
    assert result.total_triplets == 2 * 3  # 2 images × 3 splits

    for split_name in ("train", "val", "test"):
        sv = result.splits[split_name]
        assert sv.matched_triplets == 2
        assert sv.a_count == 2
        assert sv.b_count == 2
        assert sv.label_count == 2
        assert len(sv.mismatched_names) == 0
        assert len(sv.corrupt_files) == 0
        assert len(sv.missing_dirs) == 0


def test_validate_levir_cd_missing_root(tmp_path):
    """
    A non-existent root directory should return invalid with a global error.
    """
    root = tmp_path / "nonexistent_levir"
    result = validate_levir_cd(root)

    assert not result.is_valid
    assert len(result.global_errors) > 0
    # Should mention the path in the error
    assert "nonexistent_levir" in str(result.global_errors)


def test_validate_levir_cd_missing_split_dir(tmp_path):
    """
    If a split directory (e.g. val/) is missing entirely, it should be reported.
    """
    root = _build_valid_levir_root(tmp_path)
    # Remove the val split
    import shutil
    shutil.rmtree(root / "val")

    result = validate_levir_cd(root)
    assert not result.is_valid
    val_sv = result.splits["val"]
    assert len(val_sv.missing_dirs) > 0


def test_validate_levir_cd_missing_subdir(tmp_path):
    """
    If A/ is missing from a split, it should be reported as a missing dir.
    """
    root = _build_valid_levir_root(tmp_path)
    import shutil
    shutil.rmtree(root / "train" / "A")

    result = validate_levir_cd(root)
    assert not result.is_valid
    sv = result.splits["train"]
    assert any("A" in d for d in sv.missing_dirs)


def test_validate_levir_cd_mismatched_files(tmp_path):
    """
    If A/ has an extra file not present in B/ or label/, it should be reported
    as a mismatched file.
    """
    root = _build_valid_levir_root(tmp_path)
    # Add extra file to A/ only
    _create_rgb_png(root / "train" / "A" / "extra_file.png")

    result = validate_levir_cd(root)
    sv = result.splits["train"]
    # extra_file.png is in A but not B or label → mismatch
    mismatch_names = [m.split(":")[0] for m in sv.mismatched_names]
    assert "extra_file.png" in mismatch_names
    # Overall result is invalid because of mismatch
    assert not result.is_valid


def test_validate_levir_cd_corrupt_file(tmp_path):
    """
    A corrupt (empty/non-image) file in A/ should be reported as a corrupt file.
    """
    root = _build_valid_levir_root(tmp_path, names=("img_1.png",))
    # Overwrite A/img_1.png with corrupt data
    corrupt_path = root / "train" / "A" / "img_1.png"
    corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE")

    result = validate_levir_cd(root)
    sv = result.splits["train"]
    assert len(sv.corrupt_files) > 0, "Expected corrupt file to be reported"


def test_validate_levir_cd_sample_checks_binary_labels(tmp_path):
    """
    Sample checks should confirm that label masks contain only binary values {0, 255}.
    """
    root = _build_valid_levir_root(tmp_path, names=("img_1.png",))
    result = validate_levir_cd(root)

    sv = result.splits["train"]
    assert len(sv.sample_checks) > 0
    for check in sv.sample_checks:
        label_check = check["label"]
        assert label_check["ok"], f"Label check failed: {label_check.get('error')}"
        # unique_values should be subset of {0, 255}
        assert set(label_check.get("unique_values", [])).issubset({0, 255}), (
            f"Label has non-binary values: {label_check.get('unique_values')}"
        )
        assert label_check.get("is_binary", False), "Label is_binary should be True"


def test_validate_to_dict(tmp_path):
    """
    to_dict() should return a JSON-serialisable dict with all expected keys.
    """
    root = _build_valid_levir_root(tmp_path)
    result = validate_levir_cd(root)
    d = result.to_dict()

    assert "valid" in d
    assert "total_triplets" in d
    assert "splits" in d
    assert "global_errors" in d
    assert isinstance(d["splits"], dict)
    for split_name in ("train", "val", "test"):
        assert split_name in d["splits"]
        split_d = d["splits"][split_name]
        assert "a_count" in split_d
        assert "matched_triplets" in split_d
        assert "is_valid" in split_d


# ---------------------------------------------------------------------------
# LEVIRDataset tests (only if torch is available)
# ---------------------------------------------------------------------------

try:
    import torch
    import torchvision  # noqa: F401
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


@pytest.mark.skipif(not _TORCH_OK, reason="torch and torchvision not installed")
def test_levir_dataset_len(tmp_path):
    """LEVIRDataset.__len__ should equal number of matched triplets."""
    from app.services.datasets.levir_cd import LEVIRDataset

    root = _build_valid_levir_root(tmp_path, names=("img_1.png", "img_2.png", "img_3.png"))
    ds = LEVIRDataset(root, split="train", img_size=4, augment=False)
    assert len(ds) == 3, f"Expected 3, got {len(ds)}"


@pytest.mark.skipif(not _TORCH_OK, reason="torch and torchvision not installed")
def test_levir_dataset_getitem_shapes(tmp_path):
    """__getitem__ should return (img_a, img_b, label) with correct tensor shapes."""
    from app.services.datasets.levir_cd import LEVIRDataset

    # Use 8×8 images but crop size 4 (ensure crop fits)
    root = _build_valid_levir_root(tmp_path, names=("img_1.png",))
    ds = LEVIRDataset(root, split="val", img_size=4, augment=False)

    t_a, t_b, t_l = ds[0]
    assert t_a.shape == (3, 4, 4), f"img_a shape: {t_a.shape}"
    assert t_b.shape == (3, 4, 4), f"img_b shape: {t_b.shape}"
    assert t_l.shape == (1, 4, 4), f"label shape: {t_l.shape}"


@pytest.mark.skipif(not _TORCH_OK, reason="torch and torchvision not installed")
def test_levir_dataset_label_binary(tmp_path):
    """Label tensors must contain only {0.0, 1.0}, not {0, 255}."""
    from app.services.datasets.levir_cd import LEVIRDataset

    root = _build_valid_levir_root(tmp_path, names=("img_1.png",))
    ds = LEVIRDataset(root, split="train", img_size=4, augment=False)

    _, _, t_l = ds[0]
    unique_vals = torch.unique(t_l).tolist()
    assert set(unique_vals).issubset({0.0, 1.0}), (
        f"Label should be binary {{0.0, 1.0}}, got unique values: {unique_vals}"
    )


@pytest.mark.skipif(not _TORCH_OK, reason="torch and torchvision not installed")
def test_levir_dataset_image_range(tmp_path):
    """Image tensors should be in [0, 1]."""
    from app.services.datasets.levir_cd import LEVIRDataset

    root = _build_valid_levir_root(tmp_path, names=("img_1.png",))
    ds = LEVIRDataset(root, split="val", img_size=4, augment=False)

    t_a, t_b, _ = ds[0]
    assert t_a.min().item() >= 0.0 and t_a.max().item() <= 1.0
    assert t_b.min().item() >= 0.0 and t_b.max().item() <= 1.0


@pytest.mark.skipif(not _TORCH_OK, reason="torch and torchvision not installed")
def test_levir_dataset_missing_dir_raises(tmp_path):
    """LEVIRDataset should raise FileNotFoundError if the split dir does not exist."""
    from app.services.datasets.levir_cd import LEVIRDataset

    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        LEVIRDataset(root, split="train", img_size=4)


@pytest.mark.skipif(not _TORCH_OK, reason="torch and torchvision not installed")
def test_levir_dataset_invalid_split(tmp_path):
    """LEVIRDataset should raise ValueError for an invalid split name."""
    from app.services.datasets.levir_cd import LEVIRDataset

    root = _build_valid_levir_root(tmp_path, names=("img_1.png",))
    with pytest.raises(ValueError, match="split must be one of"):
        LEVIRDataset(root, split="invalid_split")
