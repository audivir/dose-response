"""Module to calculate and plot a 4PL-Dose-Response-Curve."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np
import polars as pl
import scipy.optimize as opt
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats.distributions import t
from typing_extensions import Self

from dose_response.plot import axes, errorbars, labels, plot, scatter, xticks, yticks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _typeshed import StrPath

Float64Array: TypeAlias = NDArray[np.float64]

InputT = TypeVar("InputT", float, Float64Array)


def ll4(x: InputT, hill_slope: float, bottom: float, top: float, ec50: float) -> InputT:
    """Computes response values for the fitted 4PL dose-response curve.

    Ported from the LL.4 function in the R drc package.

    Args:
        x: Dose or array of doses.
        hill_slope: Steepness of the curve.
        bottom: Lower boundary.
        top: Upper boundary.
        ec50: Relative EC50.

    Returns:
        Response for each dose in x.
    """
    return bottom + (top - bottom) / (1 + 10 ** (hill_slope * (np.log10(ec50) - np.log10(x))))  # type: ignore[no-any-return]


@dataclass
class DoseResponseCurve:
    """Stores fitted parameters of a 4PL dose-response curve.

    Y = Bottom + (Top - Bottom) / (1 + 10 ** (HillSlope * (lg(EC50) - lg(x))))
    """

    hill_slope: float
    """Steepness of the curve."""
    bottom: float
    """Lower boundary."""
    top: float
    """Upper boundary."""
    ec50: float
    """Relative EC50 value."""
    log_ec50: float = field(init=False)
    """Logarithmic EC50 in the provided unit."""
    sds: Float64Array | None = field(kw_only=True, default=None, repr=False)
    """Standard deviations of hill_slope, top, bottom, and ec50, if available."""
    sample_size: int | None = field(kw_only=True, default=None, repr=False)
    """Number of data points the curve was fitted from, required for confidence intervals."""
    alpha: float = field(kw_only=True, default=0.05, repr=False)
    """Significance level for the confidence interval."""

    def __post_init__(self) -> None:
        self.log_ec50 = np.log10(self.ec50)

        parameter_names = ("Hill Slope", "Top", "Bottom", "EC50", "LogEC50")
        means = (self.hill_slope, self.top, self.bottom, self.ec50, self.log_ec50)

        data: dict[str, Sequence[object]] = {"Parameter": parameter_names, "Mean": means}

        if self.sds is not None:
            data["SD"] = [*self.sds, None]

            if self.sample_size is not None:
                ddof = max(0, self.sample_size - len(self.sds))
                tval = float(t.ppf(1.0 - self.alpha / 2, ddof))

                ci_lower = [mean - tval * sd for mean, sd in zip(means[:-1], self.sds, strict=True)]
                ci_lower.append(np.log10(ci_lower[-1]))
                ci_upper = [mean + tval * sd for mean, sd in zip(means[:-1], self.sds, strict=True)]
                ci_upper.append(np.log10(ci_upper[-1]))

                data["CI_Lower"] = ci_lower
                data["CI_Upper"] = ci_upper

        self._params = pl.DataFrame(data)

    @property
    def params(self) -> pl.DataFrame:
        """Table of parameter names, means, and, if initialized, SD and confidence intervals."""
        return self._params


@dataclass
class DoseResponse:
    """Manages fitting and plotting a 4PL dose-response curve for one compound.

    Uses a sequence of doses and corresponding responses, fits a 4PL curve to them, and plots
    it with standard error bars.
    """

    compound: str
    doses: Float64Array
    """Doses in the provided unit. Use `DoseResponse.from_logs` first if doses are logs."""
    responses: Float64Array

    def __post_init__(self) -> None:
        if len(self.doses) != len(self.responses):
            raise ValueError("Different number of doses and responses values")

        self._log_doses = -np.log10(self.doses)

        self._data = pl.DataFrame(
            {"log_dose": self.log_doses, "dose": self.doses, "response": self.responses}
        ).sort("dose")

        self._params: DoseResponseCurve | None = None
        self._plot: Figure | None = None

    @staticmethod
    def from_logs(
        log_doses: Float64Array,
        neg: bool = True,
        log_unit: float = 1e-6,
        target_unit: float = 1e-6,
    ) -> Float64Array:
        """Converts (negative) log values in doses to a target unit.

        Args:
            log_doses: Log dose values.
            neg: Whether values are the negative log.
            log_unit: Shift of log values from the standard unit.
            target_unit: Shift of the calculated values.

        Returns:
            Array with the doses in the provided unit.
        """
        if not neg:
            log_doses *= -1

        if log_unit != target_unit:
            log_doses -= np.log10(log_unit) - np.log10(target_unit)

        return 10 ** (-log_doses)

    @property
    def log_doses(self) -> Float64Array:
        """Doses in logarithmic scale in SI."""
        return self._log_doses

    @classmethod
    def read_csv(  # noqa: PLR0913,PLR0917
        cls,
        filename: StrPath,
        compound: str | None = None,
        dose_col: int | None = None,
        response_cols: Sequence[int] | None = None,
        rm_top_rows: int = 0,
        rm_bottom_rows: int = 0,
        header: bool = False,
    ) -> DoseResponse:
        """Creates a DoseResponse instance from a CSV file.

        Values ending with `*` are excluded: for responses only the single value, for doses
        the entire row.

        Args:
            filename: Filename of the CSV file.
            compound: Compound name. `None` uses the basename of filename.
            dose_col: Index of the column with doses. `None` uses column 0.
            response_cols: Indices of the columns with responses. `None` uses every column but
                the dose column.
            rm_top_rows: Number of rows to remove from the top.
            rm_bottom_rows: Number of rows to remove from the bottom.
            header: Whether to use the first row as a header.

        Returns:
            Dose-response instance with the provided data.
        """
        filename = Path(filename)
        if compound is None:
            compound = filename.stem.upper()

        # when a header is present, the row above it is discarded.
        dr_df = pl.read_csv(
            filename, has_header=header, skip_rows=1 if header else 0, infer_schema_length=0
        )

        return cls.read_df(dr_df, compound, dose_col, response_cols, rm_top_rows, rm_bottom_rows)

    @classmethod
    def read_df(  # noqa: PLR0913,PLR0917
        cls,
        dr_df: pl.DataFrame,
        compound: str,
        dose_col: int | None = None,
        response_cols: Sequence[int] | None = None,
        rm_top_rows: int = 0,
        rm_bottom_rows: int = 0,
    ) -> Self:
        """Creates a DoseResponse instance from a DataFrame.

        Values ending with `*` are excluded: for responses only the single value, for doses
        the entire row.

        Args:
            dr_df: DataFrame with dose and response data as strings.
            compound: Compound name.
            dose_col: Index of the column with doses. `None` uses column 0.
            response_cols: Indices of the columns with responses. `None` uses every column but
                the dose column.
            rm_top_rows: Number of rows to remove from the top.
            rm_bottom_rows: Number of rows to remove from the bottom.

        Returns:
            Dose-response instance with the provided data.
        """
        if dose_col is None:
            if response_cols is not None and 0 in response_cols:
                raise ValueError(
                    "Please provide a column for doses, as default = 0 is in responses"
                )

            dose_col = 0

        if response_cols is None:
            response_cols = [i for i in range(dr_df.width) if i != dose_col]

        dr_df = cls._remove_rows(dr_df, rm_top_rows, rm_bottom_rows)

        doses = dr_df[:, dose_col]
        doses = cls._exclude_values(doses)
        doses = cls._to_numeric(doses, coerce=False)

        all_doses: list[float | None] = []
        all_responses: list[float | None] = []
        for col in response_cols:
            response_col = dr_df[:, col]
            response_col = cls._exclude_values(response_col)
            response_col = cls._to_numeric(response_col)

            all_doses.extend(doses.to_list())
            all_responses.extend(response_col.to_list())

        doses_arr, responses_arr = cls._remove_na(all_doses, all_responses)

        return cls(compound, doses_arr, responses_arr)

    @staticmethod
    def _remove_rows(clearable_df: pl.DataFrame, top: int = 0, bottom: int = 0) -> pl.DataFrame:
        """Removes rows from the top, the bottom, or both.

        Args:
            clearable_df: DataFrame to remove rows from.
            top: Number of rows to remove from the top.
            bottom: Number of rows to remove from the bottom.

        Returns:
            Cropped DataFrame.
        """
        end = clearable_df.height if bottom == 0 else -bottom
        return clearable_df[top:end]

    @staticmethod
    def _remove_na(
        doses: Sequence[float | None], responses: Sequence[float | None]
    ) -> tuple[Float64Array, Float64Array]:
        """Removes data points where the dose or response is missing or excluded.

        Args:
            doses: Dose values.
            responses: Corresponding response values.

        Returns:
            Doses and corresponding responses without missing values.
        """
        temp_df = pl.DataFrame({"dose": doses, "response": responses}).drop_nulls()
        return temp_df["dose"].to_numpy(), temp_df["response"].to_numpy()

    @staticmethod
    def _exclude_values(series: pl.Series, marker: str = "*") -> pl.Series:
        """Excludes values from data if they end with marker.

        Args:
            series: Series with data points.
            marker: Marker that marks values to exclude.

        Returns:
            Series with null instead of excluded values.
        """
        exclude = series.str.ends_with(marker).fill_null(True)
        null_series = pl.Series([None] * len(series), dtype=series.dtype)
        return null_series.zip_with(exclude, series)

    @staticmethod
    def _to_numeric(series: pl.Series, coerce: bool = True) -> pl.Series:
        """Converts series to numeric values."""
        try:
            return series.cast(pl.Float64, strict=True)
        except pl.exceptions.InvalidOperationError as e:
            if not coerce:
                raise ValueError("Strings in series, coercion turned off") from e
            print(f"Trying to convert column {series.name} with coerce mode")  # noqa: T201
            return series.cast(pl.Float64, strict=False)

    def _fit_curve(self) -> None:
        """Fits the 4PL dose-response curve with the scipy optimizer curve_fit."""
        self._fit_coefs, self._fit_pcov = opt.curve_fit(
            ll4, self.doses, self.responses, maxfev=100000
        )

    @property
    def params(self) -> DoseResponseCurve:
        """Parameters of the fitted curve."""
        if self._params is None:
            self._params = self.get_params()
        return self._params

    def get_params(self) -> DoseResponseCurve:
        """Gets the parameters of the fitted curve.

        Returns:
            The DoseResponseCurve instance with the fitted parameters.
        """
        if not hasattr(self, "_fit_coefs"):
            self._fit_curve()

        sds = np.sqrt(np.diag(self._fit_pcov))
        return DoseResponseCurve(*self._fit_coefs, sds=sds, sample_size=len(self.doses))

    @property
    def plot(self) -> Figure:
        """Plot of the fitted curve."""
        if self._plot is None:
            self._plot = self.get_plot()

        return self._plot

    def get_plot(  # noqa: PLR0913,PLR0917
        self,
        dose_unit: str = "conc. [µM]",
        response_unit: str = "fold act.",
        title: str | None = None,
        show_vals: bool = False,
        show_errorbars: bool = True,
        adjust_xticks: bool = True,
        adjust_yticks: bool = True,
    ) -> Figure:
        """Plots the fitted curve.

        Args:
            dose_unit: Label of the X-axis.
            response_unit: Label of the Y-axis.
            title: Title of the plot. `None` uses the name of the compound.
            show_vals: Whether to show all values.
            show_errorbars: Whether to show standard error bars.
            adjust_xticks: Whether to adjust ticks to logarithmic scale.
            adjust_yticks: Whether to adjust yticks to a step size of 0.5.

        Returns:
            Plot of the fitted curve.
        """
        if title is None:  # pragma: no branch
            title = self.compound.upper()

        fig = Figure()
        ax = fig.add_subplot(111)

        x_fitted, y_fitted = self._get_fitted()

        plot(ax, x_fitted, y_fitted)

        if show_vals:  # pragma: no branch
            scatter(ax, self.log_doses, self.responses)

        if show_errorbars:  # pragma: no branch
            std_x, std_y, std_err = self._get_errors()
            errorbars(ax, std_x, std_y, std_err)

        axes(ax)
        labels(ax, title, dose_unit, response_unit)

        if adjust_xticks:  # pragma: no branch
            xticks(ax, self.log_doses)
        if adjust_yticks:  # pragma: no branch
            yticks(ax, self.responses)

        self._plot = fig
        return fig

    def _get_fitted(self) -> tuple[list[float], list[float]]:
        """Gets the coordinates of the fitted curve.

        Returns:
            X values and Y values of the fitted curve.
        """
        if not hasattr(self, "_fit_coefs"):
            self._fit_curve()

        log_doses_range = np.linspace(min(self.log_doses), max(self.log_doses), 256)
        doses_range = 10**-log_doses_range

        x_fitted = list(log_doses_range)
        y_fitted = [ll4(i, *self._fit_coefs) for i in doses_range]

        return x_fitted, y_fitted

    def _get_errors(self) -> tuple[list[float], list[float], list[float]]:
        """Gets the coordinates and standard errors of the mean for the error bars.

        Returns:
            Log doses, mean responses, and standard errors of the mean per log dose.
        """
        grouped = (
            self._data.group_by("log_dose")
            .agg(
                pl.col("response").mean().alias("mean"),
                pl.col("response").std(ddof=1).alias("std"),
                pl.col("response").len().alias("size"),
            )
            .sort("log_dose")
        )

        std_err = grouped["std"] / grouped["size"].cast(pl.Float64).sqrt()

        return grouped["log_dose"].to_list(), grouped["mean"].to_list(), std_err.to_list()

    def save_plot(self, save_path: StrPath) -> None:
        """Saves the plot of the fitted curve with default settings.

        Args:
            save_path: Path to store the image file to.
        """
        self.plot.savefig(save_path)

    def save_params(self, save_path: StrPath) -> None:
        """Saves the parameters of the fitted curve as a csv file.

        Args:
            save_path: Path to store the csv file to.
        """
        self.params.params.write_csv(Path(save_path), float_precision=4)


# %%
