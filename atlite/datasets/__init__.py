# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT

"""
atlite datasets.
"""

from atlite.datasets import era5, gebco, ifs_ens, sarah

modules = {"era5": era5, "sarah": sarah, "gebco": gebco, "ifs_ens": ifs_ens}
