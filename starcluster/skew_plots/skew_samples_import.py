import numpy as np

from scipy.interpolate import RegularGridInterpolator

from .const import SAMPLES_PATH, SAMPLES_BINS

with open(SAMPLES_BINS, 'r') as f:
    NBINS = int(f.read())

mm, ss, ll, skew = np.loadtxt(SAMPLES_PATH, unpack=True)
mm, ss, ll = np.meshgrid(mm, ss, ll, indexing='ij', sparse=True)
skewness = RegularGridInterpolator((mm, ss, ll), skew, method='linear')
