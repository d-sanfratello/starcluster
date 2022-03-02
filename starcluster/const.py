import numpy as np

# Conversion constant from pc to km (IAU 2015 B2, IAU 2012 B2)
PC2KM = (648000/np.pi) * 149597870.700  # km
# Conversion from pc to kpc
PC2KPC = 1e-3
# Conversion constant from julian yr to seconds.
YR2S = 365.25 * 86400  # s
