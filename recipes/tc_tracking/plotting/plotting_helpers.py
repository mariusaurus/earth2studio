# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_handling import merge_tracks_by_time
from matplotlib.collections import LineCollection


def _make_suptitle(
    case: str,
    ic: Any = None,
    n_tracks: int | None = None,
    n_members: int | None = None,
) -> str:
    """Build a standardised plot suptitle from storm metadata."""
    title = case.split("_")[0].upper()
    if ic is not None:
        title += f"\n initialised on {ic}"
    if n_tracks is not None and n_members is not None:
        title += f"\n {n_tracks} tracks in {n_members} ensemble members"
    return title


def _var_display_info(var: str) -> tuple[str, str, float]:
    """Return ``(display_label, unit_string, scale_divisor)`` for a tracked variable."""
    info = {
        "msl": ("msl", "hPa", 100),
        "dist": ("distance", "km", 1000),
        "wind_speed": ("maximum instantaneous wind speed", "m/s", 1),
    }
    return info.get(var, (var, "", 1))


def add_some_gap(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> tuple[float, float, float, float]:
    """Expand a lat/lon bounding box by 10 % on each side and correct extreme aspect ratios.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in degrees.
    lon_min, lon_max : float
        Longitude bounds in degrees.

    Returns
    -------
    tuple[float, float, float, float]
        ``(lat_min, lat_max, lon_min, lon_max)`` with padding applied.
    """
    gap_fac = 0.1
    lat_gap = (lat_max - lat_min) * gap_fac
    lon_gap = (lon_max - lon_min) * gap_fac

    lat_min, lat_max = lat_min - lat_gap, lat_max + lat_gap
    lon_min, lon_max = lon_min - lon_gap, lon_max + lon_gap

    if lat_gap / lon_gap > 2:
        d_lon = 0.5 * (lat_max - lat_min)
        med_lon = 0.5 * (lon_min + lon_max)
        lon_min, lon_max = med_lon - d_lon / 2, med_lon + d_lon / 2

    elif lon_gap / lat_gap > 2:
        d_lat = 0.5 * (lon_max - lon_min)
        med_lat = 0.5 * (lat_min + lat_max)
        lat_min, lat_max = med_lat - d_lat / 2, med_lat + d_lat / 2

    return lat_min, lat_max, lon_min, lon_max


def get_central_coords(track: pd.DataFrame) -> tuple[float, float]:
    """Return the median latitude and longitude of a track.

    Parameters
    ----------
    track : pd.DataFrame
        Track with ``lat`` and ``lon`` columns.

    Returns
    -------
    tuple[float, float]
        ``(lat_median, lon_median)``
    """
    lat_cen = track["lat"].median()
    lon_cen = track["lon"].median()

    return lat_cen, lon_cen


def plot_spaghetti(
    true_track: pd.DataFrame,
    pred_tracks: list[dict[str, Any]],
    ensemble_mean: dict[str, Any],
    case: str,
    n_members: int,
    out_dir: str | None = None,
    alpha: float = 0.2,
    line_width: float = 2,
    ic: Any = None,
) -> None:
    """Plot ensemble track trajectories (spaghetti plot) on a map.

    Parameters
    ----------
    true_track : pd.DataFrame
        Reference track (plotted in red).
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    ensemble_mean : dict[str, Any]
        Ensemble-mean track with ``"lat"`` and ``"lon"`` arrays.
    case : str
        Storm identifier for the plot title.
    n_members : int
        Total number of ensemble members (including unmatched).
    out_dir : str or None, optional
        If provided, the figure is saved here.
    alpha : float, optional
        Transparency for ensemble member lines.
    line_width : float, optional
        Line width for all tracks.
    ic : optional
        If provided, only plot members whose ``"ic"`` is in *ic*.
    """
    plt.close("all")

    lat_cen, lon_cen = get_central_coords(true_track)

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"], len(pred_tracks), n_members),
        fontsize=16,
    )

    projection = ccrs.LambertAzimuthalEqualArea(
        central_longitude=lon_cen, central_latitude=lat_cen
    )
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.RIVERS, lw=0.5)
    if case != "debbie_2017_southern_pacific":  # cartopy issues with small islands
        ax.add_feature(cfeature.OCEAN, facecolor="#b0c4de")
        ax.add_feature(cfeature.LAND, facecolor="#C4B9A3")

    lat_min, lat_max, lon_min, lon_max = 90.0, -90.0, 360.0, -0.1

    segments = []
    for _track in pred_tracks:
        track = _track["tracks"]
        if ic is not None and _track["ic"] not in ic:
            continue

        lat_min, lat_max = min(lat_min, track["lat"].min()), max(
            lat_max, track["lat"].max()
        )
        lon_min, lon_max = min(lon_min, track["lon"].min()), max(
            lon_max, track["lon"].max()
        )

        segments.append(np.column_stack([track["lon"].values, track["lat"].values]))

    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=line_width,
                alpha=alpha,
                transform=ccrs.PlateCarree(),
            )
        )

    ax.plot(
        true_track["lon"],
        true_track["lat"],
        transform=ccrs.PlateCarree(),
        color="red",
        linewidth=line_width,
        alpha=1.0,
    )

    ax.plot(
        ensemble_mean["lon"],
        ensemble_mean["lat"],
        transform=ccrs.PlateCarree(),
        color="lime",
        linewidth=line_width,
        alpha=1.0,
    )

    lat_min, lat_max = min(lat_min, true_track["lat"].min()), max(
        lat_max, true_track["lat"].max()
    )
    lon_min, lon_max = min(lon_min, true_track["lon"].min()), max(
        lon_max, true_track["lon"].max()
    )

    lat_min, lat_max, lon_min, lon_max = add_some_gap(
        lat_min, lat_max, lon_min, lon_max
    )

    plt.tight_layout()

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)

    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{case}_tracks.png"))

    return


def normalised_intensities(
    track: pd.DataFrame, tru_track: pd.DataFrame, var: str
) -> pd.DataFrame:
    """Normalise a track variable relative to the reference track.

    For pressure (``msl``), the normalisation is
    ``(pred - ref) / (101325 - ref)``.  For other variables it is
    ``(pred - ref) / ref``.

    Parameters
    ----------
    track : pd.DataFrame
        Predicted or ensemble-mean track.
    tru_track : pd.DataFrame
        Reference track.
    var : str
        Variable name to normalise.

    Returns
    -------
    pd.DataFrame
        Merged frame with *var* replaced by its normalised values.
    """
    merged_track = merge_tracks_by_time(track, tru_track)

    if var == "msl":
        merged_track[var] = (merged_track[var] - merged_track[var + "_tru"]) / (
            101325 - merged_track[var + "_tru"]
        )
    else:
        merged_track[var] = (
            merged_track[var] - merged_track[var + "_tru"]
        ) / merged_track[var + "_tru"]

    return merged_track


def plot_relative_over_time(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    ensemble_mean: dict[str, Any],
    case: str,
    n_members: int,
    ics: Any = None,
    out_dir: str | None = None,
) -> None:
    """Plot normalised intensity deviations from the reference track over time.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    ensemble_mean : dict[str, Any]
        Ensemble statistics dict with ``"time"`` and ``"mean"`` keys.
    case : str
        Storm identifier for the plot title.
    n_members : int
        Total number of ensemble members.
    ics : optional
        If provided, only plot members whose ``"ic"`` is in *ics*.
    out_dir : str or None, optional
        If provided, the figure is saved here.
    """
    fig, _ax = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"], len(pred_tracks), n_members),
        fontsize=16,
    )

    vars = ["msl", "wind_speed"]
    labels = ["(msl - msl_ref)/(101325Pa - msl_ref)", "max_wind/max_wind_ref - 1"]

    ic, end = pred_tracks[0]["ic"], tru_track["time"].max()
    rel_steps = int(((end - ic) / np.timedelta64(6, "h") + 1) * 0.75)

    for ii in range(_ax.shape[0]):
        vmin, vmax = 1000, -1000
        for _track in pred_tracks:

            track = _track["tracks"]
            if ics is not None and _track["ic"] not in ics:
                continue

            track = normalised_intensities(track, tru_track, vars[ii])

            vmin, vmax = min(vmin, track[vars[ii]][:rel_steps].min()), max(
                vmax, track[vars[ii]][:rel_steps].max()
            )

            ax = _ax[ii]
            ax.plot(track["time"], track[vars[ii]], color="black", alpha=0.1)

        _ax[ii].set_ylabel(labels[ii])
        _ax[ii].grid(True)
        _ax[ii].set_ylim(vmin, vmax)

        ax.plot(
            tru_track["time"],
            [0 for _ in range(len(tru_track))],
            color="orangered",
            linewidth=2.5,
            label="era5 comparison",
        )

        mean = pd.DataFrame(
            {"time": ensemble_mean["time"], vars[ii]: ensemble_mean["mean"][vars[ii]]}
        )
        _track = normalised_intensities(mean, tru_track, vars[ii])
        ax.plot(
            _track["time"],
            _track[vars[ii]],
            color="lime",
            linewidth=2.5,
            label="ensemble mean",
            linestyle="--",
        )

        ax.legend()

    _ax[-1].set_xlabel("time [UTC]")

    plt.xlim(
        pred_tracks[0]["ic"] - np.timedelta64(6, "h"),
        tru_track["time"].max() + np.timedelta64(6, "h"),
    )

    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_rel_intensities.png"))

    return


def plot_over_time(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    ensemble_mean: dict[str, Any],
    case: str,
    n_members: int,
    vars: list[str] = ["msl", "wind_speed", "dist"],
    labels: list[str] | None = None,
    ics: Any = None,
    out_dir: str | None = None,
) -> None:
    """Plot absolute intensity and distance time series for all ensemble members.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    ensemble_mean : dict[str, Any]
        Ensemble statistics dict.
    case : str
        Storm identifier for the plot title.
    n_members : int
        Total number of ensemble members.
    vars : list[str]
        Variables to plot (one subplot each).
    labels : list[str] or None, optional
        Y-axis labels corresponding to *vars*.  Auto-derived from
        :func:`_var_display_info` when *None*.
    ics : optional
        If provided, only plot members whose ``"ic"`` is in *ics*.
    out_dir : str or None, optional
        If provided, the figure is saved here.
    """
    if labels is None:
        labels = [
            f"{_var_display_info(v)[0]} [{_var_display_info(v)[1]}]" for v in vars
        ]

    fig, _ax = plt.subplots(len(vars), 1, figsize=(11, 15), sharex=True)
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"], len(pred_tracks), n_members),
        fontsize=16,
    )

    t_min, t_max = np.datetime64("2120-05-16 12:00:00"), np.datetime64(
        "1820-05-16 12:00:00"
    )

    for ii in range(_ax.shape[0]):
        _, _, scale = _var_display_info(vars[ii])

        for _track in pred_tracks:
            track = _track["tracks"]
            if ics is not None and _track["ic"] not in ics:
                continue

            _ax[ii].plot(
                track["time"], track[vars[ii]] / scale, color="black", alpha=0.1
            )

            t_min, t_max = min(t_min, track["time"].min()), max(
                t_max, track["time"].max()
            )

        _ax[ii].set_xlim(t_min - np.timedelta64(6, "h"), t_max + np.timedelta64(6, "h"))
        _ax[ii].set_ylabel(labels[ii])
        _ax[ii].grid(True)

        _ax[ii].plot(
            tru_track["time"],
            tru_track[vars[ii]] / scale,
            color="orangered",
            linewidth=2.5,
            label="era5 comparison",
        )

        _ax[ii].plot(
            ensemble_mean["time"],
            ensemble_mean["mean"][vars[ii]] / scale,
            color="lime",
            linewidth=2.5,
            label="ensemble mean",
            linestyle="--",
        )
        _ax[ii].legend()

    _ax[-1].set_xlabel("time [UTC]")

    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_abs_intensities.png"))

    return


def plot_ib_era5(
    tru_track: pd.DataFrame,
    case: str,
    vars: list[str] = ["msl", "wind_speed"],
    out_dir: str | None = None,
) -> None:
    """Plot ERA5-vs-IBTrACS intensity ratios on twin y-axes.

    Parameters
    ----------
    tru_track : pd.DataFrame
        Reference track containing both ERA5 and IBTrACS columns.
    case : str
        Storm identifier for the plot title.
    vars : list[str]
        Variables to compare (``"msl"`` and/or ``"wind_speed"``).
    out_dir : str or None, optional
        If provided, the figure is saved here.
    """
    plt.close("all")

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(_make_suptitle(case), fontsize=16)

    ax2 = ax1.twinx()

    if "msl" in vars:
        p_norm = 101325
        ax1.plot(
            tru_track["time"],
            (p_norm - tru_track["msl"]) / (p_norm - tru_track["msl_ib"]),
            "black",
        )
        ax1.set_ylabel("(1013hPa-msl_era5)/(1013hPa-msl_ib)", color="black")

    if "wind_speed" in vars:
        ax2.plot(
            tru_track["time"],
            tru_track["wind_speed"] / tru_track["wind_speed_ib"],
            "orangered",
        )
        ax2.set_ylabel("wind_speed_era5/wind_speed_ib", color="orangered")

    fig.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_ib_era5_wind_speed.png"))

    return


def root_metrics(
    err_dict: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Replace MSE/variance with RMSE/standard-deviation and drop member counts.

    Parameters
    ----------
    err_dict : dict[str, dict[str, np.ndarray]]
        Per-variable error metrics (modified in place).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Updated *err_dict*.
    """
    for var in err_dict.keys():
        mse = err_dict[var].pop("mse")
        err_dict[var]["rmse"] = np.sqrt(mse)
        variance = err_dict[var].pop("variance")
        err_dict[var]["standard_deviation"] = np.sqrt(variance)
        err_dict[var].pop("n_members")

    return err_dict


def plot_errors_over_lead_time(
    err_dict: dict[str, dict[str, np.ndarray]],
    case: str,
    ic: Any,
    n_members: int,
    n_tracks: int,
    norm_dict: dict[str, float] | None = None,
    unit_dict: dict[str, str] | None = None,
    out_dir: str | None = None,
) -> None:
    """Plot error metrics (RMSE, MAE, standard deviation) as a function of lead time.

    Parameters
    ----------
    err_dict : dict[str, dict[str, np.ndarray]]
        Per-variable error metrics.
    case : str
        Storm identifier for the plot title.
    ic : Any
        Initial condition timestamp.
    n_members : int
        Total number of ensemble members.
    n_tracks : int
        Number of matched tracks.
    norm_dict : dict[str, float] or None, optional
        Normalisation divisors for display units.  Auto-derived from
        :func:`_var_display_info` when *None*.
    unit_dict : dict[str, str] or None, optional
        Display unit strings.  Auto-derived when *None*.
    out_dir : str or None, optional
        If provided, the figure is saved here.
    """
    if "mse" in err_dict[list(err_dict.keys())[0]].keys():
        err_dict = root_metrics(err_dict)

    if norm_dict is None:
        norm_dict = {v: _var_display_info(v)[2] for v in err_dict}
    if unit_dict is None:
        unit_dict = {v: _var_display_info(v)[1] for v in err_dict}

    vars = list(err_dict.keys())
    metrics = list(err_dict[vars[0]].keys())

    for extreme in ["min", "max"]:
        if extreme in metrics:
            metrics.remove(extreme)

    print(metrics)

    lead_time = np.arange(err_dict[vars[0]][metrics[0]].shape[0]) * np.timedelta64(
        6, "h"
    )

    fig, ax = plt.subplots(
        len(vars),
        len(metrics),
        figsize=((len(metrics) + 1) * 2, (len(vars) + 1) * 2),
        sharex=True,
    )

    for ivar, var in enumerate(err_dict.keys()):
        for imet, metric in enumerate(metrics):

            ax[ivar, imet].plot(lead_time, err_dict[var][metric] / norm_dict[var])

            if ivar == 0:
                ax[ivar, imet].set_title(metric, fontsize=12, weight="bold")

            if imet == 0:
                ax[ivar, imet].set_ylabel(
                    f"{var} [{unit_dict[var]}]", fontsize=12, weight="bold"
                )

            if ivar == len(vars) - 1:
                ax[ivar, imet].set_xlabel("lead time [h]", fontsize=12)

    fig.suptitle(_make_suptitle(case, ic, n_tracks, n_members), fontsize=16)

    fig.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_error_metrics_over_lead_time.png"))

    return


def extract_reference_extremes(
    tru_track: pd.DataFrame,
    pred_tracks: list[dict[str, Any]],
    ens_mean: dict[str, Any],
    vars: list[str],
) -> dict[str, dict[str, Any]]:
    """Extract per-member extreme values and the corresponding reference extremes.

    Parameters
    ----------
    tru_track : pd.DataFrame
        Reference track.
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    ens_mean : dict[str, Any]
        Ensemble statistics dict.
    vars : list[str]
        Variables to extract extremes for.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-variable dict with ``"pred"`` (array), ``"tru"`` (scalar),
        and ``"ens_mean"`` (scalar).
    """
    extreme_dict: dict[str, dict[str, Any]] = {}
    for var in vars:
        if var in ["wind_speed"]:
            reduce_fn = np.nanmax
        elif var in ["msl"]:
            reduce_fn = np.nanmin
        else:
            continue

        extreme_dict[var] = {
            "pred": np.zeros(len(pred_tracks)),
            "tru": reduce_fn(tru_track[var]),
            "ens_mean": reduce_fn(ens_mean["mean"][var]),
        }
        for ii, track in enumerate(pred_tracks):
            extreme_dict[var]["pred"][ii] = reduce_fn(track["tracks"][var])

    return extreme_dict


def add_stats_box(
    ax: plt.Axes,
    pred_var: np.ndarray,
    tru_var: float,
    var: str,
    reduction: str,
    unit: str,
) -> None:
    """Add a text box with summary statistics below a histogram axis.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to annotate.
    pred_var : np.ndarray
        Per-member extreme values.
    tru_var : float
        Reference extreme value.
    var : str
        Variable name.
    reduction : str
        ``"max"`` or ``"min"``.
    unit : str
        Display unit string.
    """
    n_exceed_spd = len(pred_var[pred_var > tru_var])
    n_total = len(pred_var)

    comp = "exceeding" if var in ["wind_speed"] else "below"
    stats = [
        ("era5 reference:", f"{tru_var:.1f} {unit}"),
        (
            f"members {comp} ref:",
            f"{n_exceed_spd} of {n_total} ({(n_exceed_spd/n_total)*100:.1f}%)",
        ),
        (f"max {reduction} {var}:", f"{pred_var.max():.1f} {unit}"),
        (f"min {reduction} {var}:", f"{pred_var.min():.1f} {unit}"),
        (f"avg {reduction} {var}:", f"{pred_var.mean():.1f} {unit}"),
        (f"std {reduction} {var}:", f"{pred_var.std():.1f} {unit}"),
    ]

    max_label_width = max(len(label) for label, _ in stats)
    text = "\n".join([f"{label:<{max_label_width}}  {value}" for label, value in stats])

    ax.text(
        0.01,
        -0.25,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        fontfamily="monospace",
    )

    return


def plot_extreme_extremes_histograms(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    ensemble_mean: dict[str, Any],
    case: str,
    vars: list[str] = ["wind_speed", "msl"],
    out_dir: str | None = None,
    nbins: int = 12,
) -> None:
    """Plot histograms of per-member extreme values with reference lines.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    ensemble_mean : dict[str, Any]
        Ensemble statistics dict.
    case : str
        Storm identifier for the plot title.
    vars : list[str]
        Variables to plot (one subplot each).
    out_dir : str or None, optional
        If provided, the figure is saved here.
    nbins : int, optional
        Number of histogram bins.
    """
    extreme_dict = extract_reference_extremes(
        tru_track, pred_tracks, ensemble_mean, vars
    )

    fig, ax = plt.subplots(1, len(vars), figsize=(3 * (len(vars) + 1), 6), sharey=True)
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"]),
        fontsize=16,
    )
    ax[0].set_ylabel("count")

    for ii, var in enumerate(vars):

        reduction = "max" if var in ["wind_speed"] else "min"
        _, unit, scale = _var_display_info(var)

        pred_var = extreme_dict[var]["pred"] / scale
        tru_var = extreme_dict[var]["tru"] / scale
        mean_var = extreme_dict[var]["ens_mean"] / scale

        ax[ii].hist(pred_var, bins=nbins)
        ax[ii].axvline(
            tru_var, color="orangered", linestyle="--", label="era5 reference"
        )
        ax[ii].axvline(mean_var, color="lime", linestyle="--", label="ensemble mean")

        ax[ii].set_title(f"{reduction} {var} (x, t)")
        ax[ii].set_xlabel(f"{var} [{unit}]")
        ax[ii].legend()

        add_stats_box(ax[ii], pred_var, tru_var, var, reduction, unit)

    fig.tight_layout()
    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{case}_histograms.png"))

    plt.show()
