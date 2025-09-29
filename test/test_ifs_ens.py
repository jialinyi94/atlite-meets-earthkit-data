import atlite
import pandas as pd

init_time = pd.Timestamp.now().normalize()
lead_time = pd.Timedelta("10D")
valid_time = init_time + lead_time
cycle = 0


def test_ifs_ens_cutout_creation():
    """Test creating cutout with IFS ENS dataset."""
    cutout = atlite.Cutout(
        path="test-ifs_ens.nc",
        module="ifs_ens",
        bounds=(-4, 56, 1.5, 62),
        time=valid_time.strftime("%Y-%m-%d"),  # the valid date of the forecast
        init_time=init_time.strftime("%Y-%m-%d"),  # the initial date of the forecast
        cycle=cycle,
    )
    assert cutout.module == ["ifs_ens"]


def test_ifs_ens_features():
    """Test IFS ENS feature availability."""
    from atlite.datasets.ifs_ens import features

    assert "wind" in features
    assert "temperature" in features
