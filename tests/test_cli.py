"""Tests for the dose_response.cli module."""

from __future__ import annotations

from pathlib import Path

from dose_response.cli import create_curve

DEMO_CSV = Path(__file__).parent / "demo" / "demo.csv"


def test_create_curve(tmp_path: Path) -> None:
    create_curve(DEMO_CSV, out=tmp_path, dose_col=0, response_cols=(1, 4))
    assert (tmp_path / "demo_plot.png").exists()
    assert (tmp_path / "demo_params.csv").exists()
