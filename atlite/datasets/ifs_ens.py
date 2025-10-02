# SPDX-FileCopyrightText: Contributors to atlite-meets-earthkit-data <https://github.com/jialinyi94/atlite-meets-earthkit-data>
#
# SPDX-License-Identifier: MIT
"""
Module for downloading and curating data from ECMWF's ensemble forecast
(via ECMWF's open-data API).

"""

import logging
import os
from pathlib import Path
from tempfile import mkstemp
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import arctan2, sqrt
from dask.utils import SerializableLock
from ecmwf.opendata import Client
import cdsapi
from numpy import atleast_1d

import atlite.datasets.era5 as era5
from atlite.pv.solar_position import SolarPosition

# Null context for running a with statements wihout any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib

    @contextlib.contextmanager
    def nullcontext():
        yield


logger = logging.getLogger(__name__)

# Ensemble forecast features
features = {
    "height": ["height"],
    "wind": ["wnd100m", "wnd_shear_exp", "wnd_azimuth"],
    "influx": [
        "influx_toa",
        "influx_direct",
        "influx_diffuse",
        "albedo",
        "solar_altitude",
        "solar_azimuth",
    ],
    "temperature": ["temperature", "soil temperature", "dewpoint temperature"],
    "runoff": ["runoff"],
}

static_features = era5.static_features

crs = era5.crs


def get_ecmwf_ifs_steps_hours(cycle: int):
    first = list(range(0, 144 + 1, 3))
    if cycle in [0, 12]:
        second = list(range(150, 360 + 1, 6))
        return first + second
    elif cycle in [6, 18]:
        return first
    
def get_cams_steps_hours(cycle: int):
    return list(range(0, 121, 1))


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and
    longitude columns as 'lat' and 'lon'.
    """
    ds = ds.rename({"longitude": "x", "latitude": "y", "valid_time": "time"})
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = era5.maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    return ds


def get_data_wind(retrieval_params):
    """
    Get wind data for given retrieval parameters.
    """
    ds = retrieve_data(
        param=[
            "10u",
            "10v",
            "100u",
            "100v",
        ],
        levtype="sfc",
        **retrieval_params,
    )
    ds = _rename_and_clean_coords(ds)

    for h in [10, 100]:
        ds[f"wnd{h}m"] = sqrt(ds[f"u{h}"] ** 2 + ds[f"v{h}"] ** 2).assign_attrs(
            units=ds[f"u{h}"].attrs["units"], long_name=f"{h} metre wind speed"
        )
    ds["wnd_shear_exp"] = (
        np.log(ds["wnd10m"] / ds["wnd100m"]) / np.log(10 / 100)
    ).assign_attrs(units="", long_name="wind shear exponent")

    # span the whole circle: 0 is north, π/2 is east, -π is south, 3π/2 is west
    azimuth = arctan2(ds["u100"], ds["v100"])
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)

    ds = ds.drop_vars(["u100", "v100", "u10", "v10", "wnd10m"])
    return ds


def sanitize_wind(ds):
    """
    Sanitize retrieved wind data. (No roughness from ECMWF Open-data API)
    """
    return ds


def get_data_influx(retrieval_params):
    """
    Get influx data for given retrieval parameters.
    """
    init_time = retrieval_params.pop("date")
    cycle = retrieval_params.pop("time")
    step = retrieval_params.pop("step")
    assert step <= 120, "CAMS data only available up to 120h lead time!"
    retrieval_params.update(
        dict(
            date=f'{init_time}/{init_time}',
            time=[f"{cycle:02d}:00"],
            leadtime_hour=[str(step)],
        )
    )
    ds = retrieve_cams_data(
        product="cams-global-atmospheric-composition-forecasts",
        variable=[
            "surface_net_solar_radiation",
            "surface_solar_radiation_downwards",
            "toa_incident_solar_radiation",
            "total_sky_direct_solar_radiation_at_surface",
        ],
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename({"fdir": "influx_direct", "tisr": "influx_toa"})
    ds["albedo"] = (
        ((ds["ssrd"] - ds["ssr"]) / ds["ssrd"].where(ds["ssrd"] != 0))
        .fillna(0.0)
        .assign_attrs(units="(0 - 1)", long_name="Albedo")
    )
    ds["influx_diffuse"] = (ds["ssrd"] - ds["influx_direct"]).assign_attrs(
        units="J m**-2", long_name="Surface diffuse solar radiation downwards"
    )
    ds = ds.drop_vars(["ssrd", "ssr"])

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a] / (60.0 * 60.0)
        ds[a].attrs["units"] = "W m**-2"

    # ERA5 variables are mean values for previous hour, i.e. 13:01 to 14:00 are labelled as "14:00"
    # account by calculating the SolarPosition for the center of the interval for aggregation happens
    # see https://github.com/PyPSA/atlite/issues/158
    # Do not show DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        time_shift = pd.to_timedelta("-30 minutes")
        sp = SolarPosition(ds, time_shift=time_shift)
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp], compat="override")

    return ds


def retrieve_data(
    model: str,
    chunks: dict[str, int] | None = None,
    tmpdir: str | Path | None = None,
    lock: SerializableLock | None = None,
    **updates,
) -> xr.Dataset:
    """
    Retrieve data from ECMWF's open-data API.

    This function retrieves data from ECMWF's open-data API using the
    `earthkit.data` package. The retrieved data is then reformatted to match
    the expected format for atlite.

    Parameters
    ----------
    model : str
        Model string for `ecmwf.opendata.Client`. See
        https://github.com/ecmwf/ecmwf-opendata#options for details.
    chunks : dict[str, int] | None
        Optional dictionary specifying the chunking of the data.
    tmpdir : str | Path | None
        Optional temporary directory for storing intermediate files.
    lock : SerializableLock | None
        Optional lock for synchronizing access to shared resources.
    **updates
        Additional keyword arguments to pass to `earthkit.data.from_source`.

    Returns
    -------
    xr.Dataset
        The retrieved and reformatted dataset.
    """
    request = dict(
        stream="enfo",
        type="pf",
    )
    request.update(updates)

    assert {"date", "time", "param", "levtype", "step"}.issubset(request), (
        "Need to specify 'date', 'time', 'param', 'levtype', and 'step' in the request"
    )

    logger.debug(f"Retrieving {model} data with request: {request}")

    client = Client(
        source="ecmwf",
        model="ifs",
        resol="0p25",
        preserve_request_order=False,
        infer_stream_keyword=True,
    )

    if lock is None:
        lock = nullcontext()

    with lock:
        fd, target = mkstemp(suffix=".grib2", dir=tmpdir)
        os.close(fd)

        # Inform user about data being downloaded as "* variable (year-month)"
        timestr = f"ForecastAt: {request['date']}T{request['time']}z, Step: {request['step']}h"
        variables = atleast_1d(request["param"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info(f"ECMWF Open-data: Downloading variables\n\t{varstr}\n")
        client.retrieve(model=model, **request, target=target)

    ds = era5.open_with_grib_conventions(
        target,
        chunks=chunks,
        tmpdir=tmpdir,
    )
    return ds


def retrieve_cams_data(
    product: str,
    chunks: dict[str, int] | None = None,
    tmpdir: str | Path | None = None,
    lock: SerializableLock | None = None,
    **updates,
) -> xr.Dataset:
    """
    Download data like CAMS from the Atmosphere Data Store (ADS).

    If you want to track the state of your request go to
    https://ads.atmosphere.copernicus.eu/requests?tab=all

    Parameters
    ----------
    product : str
        Product name, e.g. 'cams-global-atmospheric-composition-forecasts'.
    chunks : dict, optional
        Chunking for xarray dataset, e.g. {'time': 1, 'x': 100, 'y': 100}.
        Default is None.
    tmpdir : str, optional
        Directory where the downloaded data is temporarily stored.
        Default is None, which uses the system's temporary directory.
    lock : dask.utils.SerializableLock, optional
        Lock for thread-safe file writing. Default is None.
    updates : dict
        Additional parameters for the request.
        Must include 'year', 'month', and 'variable'.
        Can include e.g. 'data_format'.

    Returns
    -------
    xarray.Dataset
        Dataset with the retrieved variables.

    Examples
    --------
    >>> ds = retrieve_data(
    ...     product='cams-global-atmospheric-composition-forecasts',
    ...     chunks={'time': 1, 'x': 100, 'y': 100},
    ...     tmpdir='/tmp',
    ...     lock=None,
    ...     date='2025-10-01/2025-10-01',
    ...     time=['00:00'],
    ...     leadtime_hour=["0", "3", "6", "9", "12"],
    ...     variable=['10m_u_component_of_wind', '10m_v_component_of_wind'],
    ...     data_format='grib'
    ... )
    """
    request = {"type": ["forecast"], "data_format": "grib"}
    request.update(
        {
            k: v
            for k, v in updates.items() if k not in [
                'model',
            ]
        }
    )

    assert {"date", "time", "leadtime_hour"}.issubset(request), (
        "Need to specify at least 'date', 'time' and 'leadtime_hour' in request"
    )

    logger.info(f"Requesting {product} with API request: {request}")

    client = cdsapi.Client(
        url="https://ads.atmosphere.copernicus.eu/api",
        info_callback=logger.debug, debug=logging.DEBUG >= logging.root.level
    )
    result = client.retrieve(product, request)

    if lock is None:
        lock = nullcontext()

    suffix = f".{request['data_format']}"  # .netcdf or .grib
    with lock:
        fd, target = mkstemp(suffix=suffix, dir=tmpdir)
        os.close(fd)

        # Inform user about data being downloaded as "* variable (year-month)"
        timestr = f"{request['date']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info(f"ADS: Downloading variables\n\t{varstr}\n")
        result.download(target)

    ds = era5.open_with_grib_conventions(
        target,
        chunks=chunks,
        tmpdir=tmpdir,
    )
    return ds

sanitize_influx = era5.sanitize_influx

def get_data_temperature(retrieval_params):
    """
    Get wind temperature for given retrieval parameters.
    """
    ds = retrieve_data(
        param=[
            "2t",
            "2d",
        ],
        levtype="sfc",
        **retrieval_params,
    )

    ds_ = retrieve_data(
        param="sot",  # soil temperature
        levtype="sol",
        **retrieval_params,
    )
    ds["stl4"] = ds_["sot"].sel(soilLayer=4.0, drop=True)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename(
        {
            "t2m": "temperature",
            "stl4": "soil temperature",
            "d2m": "dewpoint temperature",
        }
    )
    return ds


def get_data_runoff(retrieval_params):
    """
    Get runoff data for given retrieval parameters.
    """
    ds = retrieve_data(param="ro", levtype="sfc", **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({"ro": "runoff"})

    return ds


def sanitize_runoff(ds):
    """
    Sanitize retrieved runoff data.
    """
    ds["runoff"] = ds["runoff"].clip(min=0.0)
    return ds


def get_data_height(retrieval_params):
    """
    Get height data for given retrieval parameters.
    """
    retrieval_params["stream"] = "oper"
    retrieval_params["type"] = "fc"
    retrieval_params["step"] = 0  # height is only available at step 0
    ds = retrieve_data(param="z", levtype="sfc", **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = era5._add_height(ds)

    return ds


def get_data(
    cutout,
    feature: str,
    tmpdir: str | Path | None,
    lock=None,
    concurrent_requests=False,
    **creation_parameters,
):
    """
    Retrieve data from ECMWF's IFS ensemble dataset.

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.ifs_ens.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    concurrent_requests : bool, optional
        If True, the monthly data requests are posted concurrently.
        Only has an effect if `monthly_requests` is True.
    **creation_parameters :
        Additional keyword arguments. The only effective argument is 'sanitize'
        (default True) which sets sanitization of the data on or off.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.

    """
    init_date = creation_parameters["init_time"]
    cycle = creation_parameters.get("cycle", 0)
    init_time = pd.Timestamp(init_date) + pd.Timedelta(hours=cycle)
    maybe_valid_times = pd.to_datetime(cutout.coords["time"].values)
    maybe_steps = ((maybe_valid_times - init_time).total_seconds() / 3600).astype(int)
    
    if feature == "influx":
        step_chunks = maybe_steps[np.isin(maybe_steps, get_cams_steps_hours(cycle))]
    else:
        step_chunks = maybe_steps[np.isin(maybe_steps, get_ecmwf_ifs_steps_hours(cycle))]

    if feature == "influx":
        assert cycle in (0, 12), (
            "CAMS data only provides forecast cycle for 00, 12 UTC"
        )
    else:
        assert cycle in (0, 6, 12, 18), (
            "ECMWF Open-data &  only provides forecast cycle for 00, 06, 12, 18 UTC"
        )

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "model": creation_parameters.get("model", "ifs"),
        "chunks": cutout.chunks,
        "tmpdir": tmpdir,
        "lock": lock,
        "date": init_date,
        "time": cycle,
    }

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    logger.info(f"Requesting data for feature {feature}...")

    def retrieve_once(step):
        ds = func({**retrieval_params, "step": step})
        if sanitize and sanitize_func is not None:
            ds = sanitize_func(ds)
        return ds

    coords = cutout.coords

    if feature in static_features:
        return (
            retrieve_once(step_chunks[0])
            .squeeze()
            .sel(
                x=slice(coords["x"].min().item(), coords["x"].max().item()),
                y=slice(coords["y"].min().item(), coords["y"].max().item()),
            )
        )

    if concurrent_requests:
        delayed_datasets = [delayed(retrieve_once)(chunk) for chunk in step_chunks]
        datasets = compute(*delayed_datasets)
    else:
        datasets = map(retrieve_once, step_chunks)

    ds = xr.concat(datasets, dim="time")

    ds = ds.sel(
        x=slice(coords["x"].min().item(), coords["x"].max().item()),
        y=slice(coords["y"].min().item(), coords["y"].max().item()),
    )
    return ds


if __name__ == "__main__":
    ds = get_data_influx(
        retrieval_params=dict(
            date='2025-10-01/2025-10-01',
            time=['00:00'],
            leadtime_hour=["0", "1", "2"],
        )
    )
    print(ds)
