"""Command-line interface for the Dose-Response Curve package."""

from __future__ import annotations

from pathlib import Path

import doctyper

from dose_response.drc import DoseResponse


def create_curve(
    file: Path,
    out: Path = Path(),
    dose_col: int | None = None,
    response_cols: tuple[int, int] | None = None,
    header: bool = False,
) -> None:
    """Creates a dose-response curve from a CSV file.

    Args:
        file: CSV file to load dose and response data from.
        out: Directory to store the plot and parameters to.
        dose_col: Column index of doses.
        response_cols: Start and end column index of the responses.
        header: Whether to use the first row as a header.
    """
    output_dir = out.expanduser().resolve()
    resolved_response_cols = (
        list(range(response_cols[0], response_cols[1] + 1)) if response_cols is not None else None
    )

    dr = DoseResponse.read_csv(
        file, dose_col=dose_col, response_cols=resolved_response_cols, header=header
    )

    lower_compound = dr.compound.lower().replace(" ", "_")
    dr.save_plot(output_dir / f"{lower_compound}_plot.png")
    dr.save_params(output_dir / f"{lower_compound}_params.csv")


def main() -> None:  # pragma: no cover
    """Runs the `drc` console script."""
    app = doctyper.DocTyper()
    app.command()(create_curve)
    app()


if __name__ == "__main__":
    main()
