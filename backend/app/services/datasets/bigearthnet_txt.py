"""
backend/app/services/datasets/bigearthnet_txt.py
=================================================
BigEarthNet annotation parquet summary service.

IMPORTANT: BigEarthNet.txt.parquet contains text-based question-answer annotations
(VQA-style), NOT image pixel data. The actual Sentinel-1/Sentinel-2 image bands
are NOT included in these files and are NOT downloaded by this service.

File description:
    BigEarthNet.txt.parquet — 9,553,962 rows of VQA annotations
        Columns: ID, s1_name, patch_id, input (question), output (answer),
                 type, category, split, latitude, longitude, country, season,
                 climate_zone

    metadata.parquet — 480,038 patch-level metadata rows
        Columns: patch_id, labels (list of land-cover classes), split, country,
                 s1_name, s2v1_name, contains_seasonal_snow, contains_cloud_or_shadow

This module reads only schema + a sample of rows using pyarrow to inspect the
files without loading them fully into RAM. It gracefully handles:
  - Missing pyarrow dependency
  - Missing or corrupt parquet files
  - Any other I/O error
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...logging_setup import logger


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ParquetFileSummary:
    path: str
    available: bool
    error: Optional[str] = None
    row_count: Optional[int] = None
    column_names: Optional[List[str]] = None
    schema_str: Optional[str] = None
    sample_rows: Optional[List[Dict[str, Any]]] = None
    distributions: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "available": self.available,
            "error": self.error,
            "row_count": self.row_count,
            "column_names": self.column_names,
            "schema_str": self.schema_str,
            "sample_rows": self.sample_rows,
            "distributions": self.distributions,
        }


@dataclass
class BigEarthNetSummary:
    txt_annotation: ParquetFileSummary
    metadata: ParquetFileSummary
    pyarrow_available: bool
    note: str = (
        "BigEarthNet.txt.parquet contains VQA text annotations, NOT image pixel data. "
        "The Sentinel-1/Sentinel-2 image bands are separate and are NOT downloaded here."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note": self.note,
            "pyarrow_available": self.pyarrow_available,
            "txt_annotation": self.txt_annotation.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _summarise_parquet(path: Path, sample_n: int = 3) -> ParquetFileSummary:
    """
    Read schema + a small sample from a parquet file using pyarrow.
    Does NOT load all rows into memory.
    """
    if not path.exists():
        return ParquetFileSummary(
            path=str(path),
            available=False,
            error=f"File not found: {path}",
        )

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ParquetFileSummary(
            path=str(path),
            available=False,
            error="pyarrow is not installed. Install via: pip install pyarrow",
        )

    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        row_count = pf.metadata.num_rows
        col_names = schema.names

        # Read only the first batch (memory-safe)
        first_batch = next(pf.iter_batches(batch_size=sample_n))
        batch_table = first_batch.to_pydict()
        sample_rows = [
            {col: batch_table[col][i] for col in col_names}
            for i in range(min(sample_n, len(batch_table[col_names[0]])))
        ]

        # Compute split / type / category distributions (if those columns exist)
        distributions: Dict[str, Any] = {}
        dist_cols = [c for c in ("split", "type", "category", "country", "season") if c in col_names]
        if dist_cols:
            try:
                # Read just the distribution columns — much smaller than full file
                dist_table = pq.read_table(path, columns=dist_cols)
                for col in dist_cols:
                    try:
                        import pyarrow.compute as pc
                        vals = dist_table.column(col)
                        # value_counts returns a struct array
                        vc = pc.value_counts(vals)
                        vc_dict = {
                            str(item["values"].as_py()): int(item["counts"].as_py())
                            for item in vc.to_pylist()
                        }
                        # Sort by count descending, cap at top-10
                        distributions[col] = dict(
                            sorted(vc_dict.items(), key=lambda x: -x[1])[:10]
                        )
                    except Exception as dist_err:
                        distributions[col] = {"error": str(dist_err)}
            except Exception as dist_err:
                logger.warning("[bigearthnet] Distribution computation failed: %s", dist_err)

        return ParquetFileSummary(
            path=str(path),
            available=True,
            row_count=row_count,
            column_names=col_names,
            schema_str=str(schema),
            sample_rows=sample_rows,
            distributions=distributions if distributions else None,
        )

    except Exception as e:
        logger.warning("[bigearthnet] Failed to read parquet %s: %s", path, e)
        return ParquetFileSummary(
            path=str(path),
            available=False,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_bigearthnet_summary(
    txt_path: Optional[Path],
    meta_path: Optional[Path],
) -> BigEarthNetSummary:
    """
    Summarise BigEarthNet annotation and metadata parquet files.

    Reads only schema + a tiny sample of each file. Does NOT load all rows
    into memory. Safely handles missing pyarrow or missing files.

    Parameters
    ----------
    txt_path  : Path to BigEarthNet.txt.parquet (may be None if not configured).
    meta_path : Path to metadata.parquet (may be None if not configured).

    Returns
    -------
    BigEarthNetSummary with per-file ParquetFileSummary objects.
    """
    pyarrow_ok = _check_pyarrow()

    if txt_path is None:
        txt_summary = ParquetFileSummary(
            path="(not configured)",
            available=False,
            error="BIGEARTHNET_TXT_PARQUET environment variable is not set.",
        )
    else:
        txt_summary = _summarise_parquet(txt_path)

    if meta_path is None:
        meta_summary = ParquetFileSummary(
            path="(not configured)",
            available=False,
            error="BIGEARTHNET_METADATA_PARQUET environment variable is not set.",
        )
    else:
        meta_summary = _summarise_parquet(meta_path)

    logger.info(
        "[bigearthnet] Summary complete. txt_available=%s meta_available=%s pyarrow=%s",
        txt_summary.available,
        meta_summary.available,
        pyarrow_ok,
    )
    return BigEarthNetSummary(
        txt_annotation=txt_summary,
        metadata=meta_summary,
        pyarrow_available=pyarrow_ok,
    )
