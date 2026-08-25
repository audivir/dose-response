"""Tests for the dose_response.generate_test_data module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dose_response.generate_test_data import generate_test_data

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_test_data_writes_csv(tmp_path: Path) -> None:
    out_file = tmp_path / "test_data.csv"
    generate_test_data(out_file)
    written = pl.read_csv(out_file, has_header=False)
    assert written.shape == (10, 6)
