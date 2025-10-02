# SPDX-FileCopyrightText: Contributors to atlite-meets-earthkit-data <https://github.com/jialinyi94/atlite-meets-earthkit-data>
#
# SPDX-License-Identifier: MIT

import pandas as pd

import atlite

init_time = pd.Timestamp.now().normalize() - pd.Timedelta("1D") # make sure init_time is always querable
lead_time = pd.Timedelta("4D") # lead time within available range of IFS ENS and CAMS data
valid_time = init_time + lead_time
cycle = 0

cutout = atlite.Cutout(
    path="test-ifs_ens.nc",
    module="ifs_ens",
    bounds=(-4, 56, 1.5, 62),
    time=valid_time.strftime("%Y-%m-%d"),  # the valid date of the forecast
    init_time=init_time.strftime("%Y-%m-%d"),  # the initial date of the forecast
    cycle=cycle,
)

def test_ifs_ens_cutout_creation():
    """Test creating cutout with IFS ENS dataset."""
    assert cutout.module == ["ifs_ens"]


def test_ifs_ens_features():
    """Test IFS ENS feature availability."""
    from atlite.datasets.ifs_ens import features

    assert "wind" in features
    assert "temperature" in features

def test_ifs_ens_influx_data():
    cutout.prepare("influx", show_progress=True)
    ds = cutout.data
    assert "influx_toa" in ds