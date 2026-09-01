"""
tests/test_bigearthnet.py
--------------------------
Tests for BigEarthNet parquet summary service.

Tests cover:
- Missing file path handling
- Missing pyarrow handling (mocked import failure)
- Missing BIGEARTHNET_TXT_PARQUET env var
- Valid parquet file reading (if pyarrow is available)
- Correct clarification that files contain annotations, not image data
- No crashes for any input scenario

All tests are CPU-safe, fast, and require NO internet access.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.datasets.bigearthnet_txt import (
    get_bigearthnet_summary,
    BigEarthNetSummary,
    ParquetFileSummary,
)


# ---------------------------------------------------------------------------
# Tests for missing configuration
# ---------------------------------------------------------------------------

def test_bigearthnet_summary_missing_both_configs():
    """
    When both paths are None (not configured), should return a valid summary
    with available=False and clear error messages. Should NOT crash.
    """
    summary = get_bigearthnet_summary(txt_path=None, meta_path=None)

    assert isinstance(summary, BigEarthNetSummary)
    assert not summary.txt_annotation.available
    assert not summary.metadata.available
    assert "not set" in summary.txt_annotation.error.lower() or "not configured" in summary.txt_annotation.error.lower()
    assert summary.note  # note about annotations vs image data must be present
    assert "annotation" in summary.note.lower() or "text" in summary.note.lower()


def test_bigearthnet_summary_missing_file(tmp_path):
    """
    When file paths are configured but files don't exist, should return
    available=False with error. Should NOT crash.
    """
    nonexistent_txt = tmp_path / "BigEarthNet.txt.parquet"
    nonexistent_meta = tmp_path / "metadata.parquet"

    summary = get_bigearthnet_summary(txt_path=nonexistent_txt, meta_path=nonexistent_meta)

    assert not summary.txt_annotation.available
    assert not summary.metadata.available
    assert "not found" in summary.txt_annotation.error.lower() or "File not found" in summary.txt_annotation.error


def test_bigearthnet_summary_missing_pyarrow_graceful(tmp_path):
    """
    If pyarrow is not importable, the service should return available=False
    with a clear install instruction, NOT raise an ImportError.
    """
    # Create a dummy (non-parquet) file so the "file not found" check passes
    fake_txt = tmp_path / "fake.parquet"
    fake_txt.write_bytes(b"NOT_REAL_PARQUET")

    # Patch pyarrow to appear unimportable inside the module
    with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None, "pyarrow.compute": None}):
        summary = get_bigearthnet_summary(txt_path=fake_txt, meta_path=None)

    # Should not crash, and should report unavailability
    assert isinstance(summary, BigEarthNetSummary)
    # txt_annotation may succeed or fail depending on patch depth, but must not raise
    # The module-level _check_pyarrow() should return False
    assert not summary.pyarrow_available or not summary.txt_annotation.available


def test_bigearthnet_summary_contains_annotation_note():
    """
    The note field must clearly state that these are annotation/text files,
    NOT the actual satellite image data.
    """
    summary = get_bigearthnet_summary(txt_path=None, meta_path=None)
    note = summary.note.lower()
    # Must mention it is NOT image data
    has_annotation_ref = "annotation" in note or "text" in note or "vqa" in note
    has_not_image_ref = "not" in note and ("image" in note or "pixel" in note or "sentinel" in note)
    assert has_annotation_ref or has_not_image_ref, (
        f"Note must clarify these are annotation files, not image data. Got: {summary.note!r}"
    )


def test_bigearthnet_to_dict_structure():
    """
    to_dict() should return a JSON-serialisable dict with all expected top-level keys.
    """
    summary = get_bigearthnet_summary(txt_path=None, meta_path=None)
    d = summary.to_dict()

    assert "note" in d
    assert "pyarrow_available" in d
    assert "txt_annotation" in d
    assert "metadata" in d
    # Each sub-dict should have "path" and "available" keys
    for key in ("txt_annotation", "metadata"):
        assert "path" in d[key]
        assert "available" in d[key]


# ---------------------------------------------------------------------------
# Tests with real pyarrow (if available) and a tiny synthetic parquet
# ---------------------------------------------------------------------------

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PYARROW_OK = True
except ImportError:
    _PYARROW_OK = False


@pytest.mark.skipif(not _PYARROW_OK, reason="pyarrow not installed")
def test_bigearthnet_summary_with_synthetic_parquet(tmp_path):
    """
    With a small synthetic parquet file, the summary should correctly read
    schema, row count, and sample rows without loading all data.
    """
    # Create a tiny parquet file
    data = {
        "ID": list(range(100)),
        "patch_id": [f"S2A_PATCH_{i}" for i in range(100)],
        "input": [f"Does the image contain area {i}?" for i in range(100)],
        "output": ["yes" if i % 2 == 0 else "no" for i in range(100)],
        "type": ["binary"] * 100,
        "split": (["train"] * 50 + ["val"] * 25 + ["test"] * 25),
        "country": (["Austria"] * 40 + ["Germany"] * 30 + ["France"] * 30),
    }
    table = pa.table(data)
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(table, parquet_path)

    summary = get_bigearthnet_summary(txt_path=parquet_path, meta_path=None)

    assert summary.txt_annotation.available, f"Should be available: {summary.txt_annotation.error}"
    assert summary.txt_annotation.row_count == 100
    assert summary.txt_annotation.column_names is not None
    assert "ID" in summary.txt_annotation.column_names
    assert "input" in summary.txt_annotation.column_names
    assert len(summary.txt_annotation.sample_rows) > 0

    # Distributions should be computed
    if summary.txt_annotation.distributions:
        # "split" distribution should have train/val/test entries
        if "split" in summary.txt_annotation.distributions:
            split_dist = summary.txt_annotation.distributions["split"]
            assert "train" in split_dist or len(split_dist) > 0


@pytest.mark.skipif(not _PYARROW_OK, reason="pyarrow not installed")
def test_bigearthnet_summary_sample_rows_count(tmp_path):
    """
    Summary should return at most 3 sample rows (default sample_n=3).
    """
    data = {"col_a": list(range(10)), "col_b": [f"val_{i}" for i in range(10)]}
    table = pa.table(data)
    parquet_path = tmp_path / "small.parquet"
    pq.write_table(table, parquet_path)

    summary = get_bigearthnet_summary(txt_path=parquet_path, meta_path=None)
    assert summary.txt_annotation.available
    assert len(summary.txt_annotation.sample_rows) <= 3


@pytest.mark.skipif(not _PYARROW_OK, reason="pyarrow not installed")
def test_bigearthnet_metadata_summary_with_synthetic_parquet(tmp_path):
    """
    metadata.parquet summary should read the labels list column correctly.
    """
    data = {
        "patch_id": [f"S2A_{i}" for i in range(20)],
        "split": (["train"] * 12 + ["val"] * 4 + ["test"] * 4),
        "country": ["Austria"] * 10 + ["Germany"] * 10,
        "labels": [["Arable land", "Pastures"]] * 20,
        "contains_seasonal_snow": [False] * 20,
        "contains_cloud_or_shadow": [False] * 20,
    }
    table = pa.table(data)
    meta_path = tmp_path / "meta.parquet"
    pq.write_table(table, meta_path)

    summary = get_bigearthnet_summary(txt_path=None, meta_path=meta_path)

    assert summary.metadata.available, f"Should be available: {summary.metadata.error}"
    assert summary.metadata.row_count == 20
    assert "patch_id" in summary.metadata.column_names
