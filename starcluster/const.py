import numpy as np

# Conversion constant from pc to km (IAU 2015 B2, IAU 2012 B2)
PC2KM = (648000/np.pi) * 149597870.700  # km
# Conversion from pc to kpc
PC2KPC = 1e-3
# Conversion constant from julian yr to seconds.
YR2S = 365.25 * 86400  # s

# The transformation matrix from ICRS coordinates to galactic coordinates,
# named as the inverse matrix from galactic to ICRS coordinates. The exact
# values are as defined in Hobbs et al., 2021, Ch.4.
A_G_INV = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])
