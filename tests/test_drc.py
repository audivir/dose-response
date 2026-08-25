"""Tests for the dose_response.drc module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from dose_response.drc import DoseResponse, DoseResponseCurve, ll4

if TYPE_CHECKING:
    from dose_response.drc import Float64Array

DEMO_DIR = Path(__file__).parent / "demo"
DEMO_CSV = DEMO_DIR / "demo.csv"
DEMO_PARAMS_CSV = DEMO_DIR / "demo_params.csv"


def test_ll4() -> None:
    response = ll4(1e-6, hill_slope=1.0, bottom=1.0, top=3.0, ec50=1e-6)
    assert isinstance(response, float)
    assert response == pytest.approx(2.0)

    x: Float64Array = np.array([1e-7, 1e-6, 1e-5])
    array_response = ll4(x, hill_slope=1.0, bottom=1.0, top=3.0, ec50=1e-6)
    assert isinstance(array_response, np.ndarray)


def test_dose_response_curve() -> None:
    curve = DoseResponseCurve(hill_slope=1.0, bottom=1.0, top=3.0, ec50=1e-6)
    assert curve.log_ec50 == pytest.approx(-6.0)
    assert curve.params.columns == ["Parameter", "Mean"]

    sds = np.array([0.1, 0.1, 0.1, 1e-7])
    curve = DoseResponseCurve(hill_slope=1.0, bottom=1.0, top=3.0, ec50=1e-6, sds=sds)
    assert curve.params.columns == ["Parameter", "Mean", "SD"]
    assert curve.params["SD"][-1] is None

    sds = np.array([0.1, 0.1, 0.1, 1e-7])
    curve = DoseResponseCurve(
        hill_slope=1.0, bottom=1.0, top=3.0, ec50=1e-6, sds=sds, sample_size=10
    )
    assert curve.params.columns == ["Parameter", "Mean", "SD", "CI_Lower", "CI_Upper"]
    assert curve.params["CI_Lower"][-1] is not None

    with pytest.raises(ValueError, match="Different number"):
        DoseResponse("A", np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize(
    ("neg", "log_unit", "target_unit"),
    [(True, 1e-6, 1e-6), (False, 1e-6, 1e-6), (True, 1e-6, 1e-3)],
)
def test_from_logs_returns_positive_doses(neg: bool, log_unit: float, target_unit: float) -> None:
    log_doses = np.array([6.0, 5.0]) if neg else np.array([-6.0, -5.0])
    doses = DoseResponse.from_logs(log_doses, neg=neg, log_unit=log_unit, target_unit=target_unit)
    assert (doses > 0).all()


def test_read_csv(tmp_path: Path) -> None:
    # infers col
    dr = DoseResponse.read_csv(DEMO_CSV)
    assert dr.compound == "DEMO"
    assert len(dr.doses) == 35  # 10 rows * 4 responses, minus 5 marked with "*"

    dr = DoseResponse.read_csv(DEMO_CSV, compound="Test", dose_col=0, response_cols=[1, 2, 3, 4])
    assert dr.compound == "Test"
    assert len(dr.doses) == 35

    header_csv = tmp_path / "demo_with_header.csv"
    header_csv.write_text(
        "throwaway line\ndose,resp1,resp2\n0.01,1.0,1.1\n0.1,1.5,1.6\n1.0,2.0,2.1\n"
    )
    dr = DoseResponse.read_csv(header_csv, header=True)
    assert len(dr.doses) == 6


def test_read_df(capsys: pytest.CaptureFixture[str]) -> None:
    df = pl.DataFrame({"a": ["0.01"], "b": ["1.0"]})
    with pytest.raises(ValueError, match="Please provide a column for doses"):
        DoseResponse.read_df(df, "TEST", response_cols=[0, 1])

    # skip
    df = pl.DataFrame(
        {"dose": ["skip", "0.01", "0.1", "skip"], "resp": ["skip", "1.0", "2.0", "skip"]}
    )
    dr = DoseResponse.read_df(
        df, "TEST", dose_col=0, response_cols=[1], rm_top_rows=1, rm_bottom_rows=1
    )
    assert len(dr.doses) == 2

    # missing doses
    df = pl.DataFrame({"dose": ["0.01", "0.1", "1.0"], "resp": ["1.0", None, "2.0"]})
    dr = DoseResponse.read_df(df, "TEST", dose_col=0, response_cols=[1])
    assert len(dr.doses) == 2

    # non-numeric dose
    df = pl.DataFrame({"dose": ["0.01", "bad"], "resp": ["1.0", "2.0"]})
    with pytest.raises(ValueError, match="Strings in series"):
        DoseResponse.read_df(df, "TEST", dose_col=0, response_cols=[1])

    df = pl.DataFrame({"dose": ["0.01", "0.1"], "resp": ["1.0", "bad"]})
    dr = DoseResponse.read_df(df, "TEST", dose_col=0, response_cols=[1])
    assert len(dr.doses) == 1
    assert "coerce mode" in capsys.readouterr().out


def test_get_params_matches_reference() -> None:
    dr = DoseResponse.read_csv(DEMO_CSV, dose_col=0, response_cols=[1, 2, 3, 4])
    reference = pl.read_csv(DEMO_PARAMS_CSV)
    for row, ref_row in zip(
        dr.params.params.iter_rows(named=True), reference.iter_rows(named=True), strict=True
    ):
        assert row["Parameter"] == ref_row["Parameter"]
        assert row["Mean"] == pytest.approx(ref_row["Mean"], abs=1e-3)


def test_get_plot_and_save(tmp_path: Path) -> None:
    dr = DoseResponse.read_csv(DEMO_CSV, dose_col=0, response_cols=[1, 2, 3, 4])

    _ = dr.params  # fits the curve lazily, before get_plot re-uses the same fit

    fig1 = dr.get_plot(show_vals=True)
    fig2 = dr.plot
    assert fig1 is fig2

    plot_path = tmp_path / "plot.png"
    params_path = tmp_path / "params.csv"
    dr.save_plot(plot_path)
    dr.save_params(params_path)
    assert plot_path.exists()
    assert params_path.exists()
