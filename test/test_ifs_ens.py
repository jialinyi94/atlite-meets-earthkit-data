import pytest
import atlite

def test_ifs_ens_cutout_creation():
    """Test creating cutout with IFS ENS dataset."""
    cutout = atlite.Cutout(
        path="test-ifs_ens.nc",
        module="ifs_ens",
        bounds=(-4, 56, 1.5, 62),
        time="2023-01-01",
    )
    assert cutout.module == ["ifs_ens"]

def test_ifs_ens_features():
    """Test IFS ENS feature availability."""
    from atlite.datasets.ifs_ens import features
    assert "wind" in features
    assert "temperature" in features
