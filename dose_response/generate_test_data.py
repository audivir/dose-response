"""Generates synthetic dose and response test data for the DRC package."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import doctyper
import numpy as np
import polars as pl

from dose_response.drc import ll4

if TYPE_CHECKING:
    from numpy.typing import NDArray


def generate_test_data(out: Path) -> None:
    """Generates test dose and response values and saves them to a CSV file.

    Args:
        out: File to save the test data to.
    """
    hill_slope, bottom, top, ec50, duplicates, unit = 1.0, 1.0, 3.0, 1e-6, 5, 1e-6
    base_concs = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
    concs = np.array(base_concs, dtype=np.float64)

    rng = np.random.default_rng()
    responses: list[NDArray[np.float64]] = []
    for conc in concs * unit:
        col = duplicates * [ll4(conc, hill_slope, bottom, top, ec50)]
        noise = rng.normal(1.5, 0.1, size=(duplicates,))
        responses.append(noise * col)

    stacked_responses = np.stack(responses)
    concs_n_responses = np.insert(stacked_responses, 0, concs, axis=1)

    dr_df = pl.DataFrame(concs_n_responses)
    dr_df.write_csv(out, include_header=False, float_precision=4)


def main() -> None:  # pragma: no cover
    """Runs the `drc-test` console script."""
    app = doctyper.DocTyper()
    app.command()(generate_test_data)
    app()


if __name__ == "__main__":
    main()
