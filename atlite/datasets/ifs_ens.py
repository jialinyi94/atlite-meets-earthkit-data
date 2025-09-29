"""
Module for downloading and curating data from ECMWF's ensemble forecast
(via ECMWF's open-data API).

"""
import logging
import os
from pathlib import Path
from tempfile import mkstemp
from numpy import atleast_1d
from dask import compute, delayed
from dask.utils import SerializableLock
from dask.array import sqrt, arctan2
import numpy as np
import pandas as pd
import xarray as xr
import atlite
import atlite.datasets.era5 as era5
from ecmwf.opendata import Client

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
        "ssrd",
        "albedo",
    ],
    "temperature": ["temperature", "soil temperature", "dewpoint temperature"],
    "runoff": ["runoff"],
}

static_features = era5.static_features

crs = era5.crs


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and
    longitude columns as 'lat' and 'lon'.
    """
    ds = (
        ds
        .rename({"longitude": "x", "latitude": "y", "valid_time": "time"})
    )
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
            "10u", "10v", "100u", "100v",
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
    ds = retrieve_data(
        param=["ssrd", "ssr"],
        levtype="sfc",
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)
    ds["albedo"] = (
        ((ds["ssrd"] - ds["ssr"]) / ds["ssrd"].where(ds["ssrd"] != 0))
        .fillna(0.0)
        .assign_attrs(units="(0 - 1)", long_name="Albedo")
    )
    ds = ds.drop_vars("ssr")
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
        client.retrieve(model=model,**request, target=target)

    ds = era5.open_with_grib_conventions(
        target,
        chunks=chunks,
        tmpdir=tmpdir,
    )
    return ds


def sanitize_influx(ds):
    """
    Sanitize retrieved influx data.
    """
    for a in ("ssrd", "albedo"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def get_data_temperature(retrieval_params):
    """
    Get wind temperature for given retrieval parameters.
    """
    ds = retrieve_data(
        param=[
            "2t", "2d",
        ],
        levtype="sfc",
        **retrieval_params,
    )

    ds_ = retrieve_data(
        param="sot", # soil temperature
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
    ds = retrieve_data(
        param="ro",
        levtype="sfc",
        **retrieval_params
    )

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
    ds = retrieve_data(
        param="z", 
        levtype="sfc",
        **retrieval_params
    )

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
    assert "step" in creation_parameters, (
        "Need to specify 'step' in cutout creation parameters"
    )
    step = creation_parameters["step"]
    if isinstance(step, int):
        step_chunks = [step]
    else:
        step_chunks = step
    forecast_time = pd.Timestamp(cutout.coords["time"].item())
    time = forecast_time.hour
    assert time in (0, 6, 12, 18), "ECMWF Open-data only provides forecasts for 00, 06, 12, 18 UTC"

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "model": creation_parameters.get("model", "ifs"),
        "chunks": cutout.chunks,
        "tmpdir": tmpdir,
        "lock": lock,
        "date": forecast_time.date(),
        "time": time,
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
        return retrieve_once(step_chunks[0]).squeeze().sel(
            x=slice(coords["x"].min().item(), coords["x"].max().item()),
            y=slice(coords["y"].min().item(), coords["y"].max().item()),
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
    # Example usage
    cutout = atlite.Cutout(
        path="test_ifs_ens.nc",
        module="ifs_ens",
        x=slice(-13.6913, 1.7712),
        y=slice(49.9096, 60.8479),
        time="2025-09-28 06:00",
    )
    create_parameters = dict(
        step=[0, 18],
        sanitize=True,
    )

    ds = get_data(
        cutout,
        feature="influx",
        tmpdir=None,
        concurrent_requests=False,
        **create_parameters,
    )
    print(ds)
