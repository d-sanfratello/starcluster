import os

from numpy import pi
from pathlib import Path

# FIXME: check for real constants value and add references to them
R_GAL = 15  # kpc
R_S_SAGA = 8  # kpc

PAR_MIN = 1e-6  # mas
PAR_MAX = 1e3  # mas
PAR_ERR_MIN = 0.  # mas
PAR_ERR_MAX = 2.  # mas
GAL_LONG_MIN = 0.  # rad
GAL_LONG_MAX = 2 * pi  # rad

SAMPLES_PATH = Path(os.getcwd()).joinpath(
    './starcluster/skew_plots/samples_data/skew_samples.csv')
SAMPLES_BINS = Path(os.getcwd()).joinpath(
    './starcluster/skew_plots/samples_data/nbins.txt')
