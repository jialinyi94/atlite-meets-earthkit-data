"""
Module for downloading and curating data from ECMWF's ensemble forecast
(via ECMWF's open-data API).

"""
import logging
import os
from pathlib import Path
from tempfile import mkstemp
from numpy import atleast_1d
from dask.utils import SerializableLock
from dask.array import sqrt, arctan2
import numpy as np
import xarray as xr
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


def get_data_wind(retrieval_params):
    """
    Get wind data for given retrieval parameters.
    """
    ds = retrieve_data(
        param=[
            "10u", "10v", "100u", "100v"
        ],
        **retrieval_params,
    )
    ds = era5.rename_and_clean_coords(ds)

    for h in [10, 100]:
        ds[f"wnd{h}m"] = sqrt(ds[f"{h}u"] ** 2 + ds[f"{h}v"] ** 2).assign_attrs(
            units=ds[f"{h}u"].attrs["units"], long_name=f"{h} metre wind speed"
        )
    ds["wnd_shear_exp"] = (
        np.log(ds["wnd10m"] / ds["wnd100m"]) / np.log(10 / 100)
    ).assign_attrs(units="", long_name="wind shear exponent")

    # span the whole circle: 0 is north, π/2 is east, -π is south, 3π/2 is west
    azimuth = arctan2(ds["100u"], ds["100v"])
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)

    ds = ds.drop_vars(["100u", "100v", "10u", "10v", "wnd10m"])
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
        timestr = f"{request['year']}-{request['month']}"
        variables = atleast_1d(request["variable"])
        varstr = "\n\t".join([f"{v} ({timestr})" for v in variables])
        logger.info(f"ECMWF Open-data: Downloading variables\n\t{varstr}\n")
        client.retrieve(model=model,**request, target=target)
    
    ds = xr.open_dataset(target, decode_timedelta=True, engine="cfgrib")
    if tmpdir is None:
        era5.add_finalizer(target)
    
    return ds

