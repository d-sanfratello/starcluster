import numpy as np

from starcluster.const import A_G_INV


keys = ['ra', 'dec', 'l', 'b', 'parallax',
        'pmra', 'pmdec', 'radial_velocity']
mags = ['g_mag', 'bp_mag', 'rp_mag',
        'bp_rp', 'bp_g', 'g_rp']
data = ['l', 'b', 'parallax', 'pml', 'pmb', 'radial_velocity']
save_names = list(data)
save_names.extend(['mag', 'c_index'])
save_dtype = np.dtype([
          (name, float)
          for name in save_names
      ])


class ExpectedValues:
    def __init__(self, expected, from_file=False):
        """
        Class to interpret the expected values for a cluster for the quantities
        used with the DPGMM, from the equatorial coordinates and relative
        proper motions, parallax, radial velocity and galactic latitude and
        longitude.

        Right ascension, declination, galactic longitude and latitude,
        parallax, radial_velocity, and proper motions in ra and dec are
        mandatory keys, but photometric band and color index can be omitted
        and will be set to `np.nan`.

        Parameters
        ----------
        expected:
            'dictionary'. The expected keys are 'ra', 'dec', 'pmra', 'pmdec',
            'parallax', 'radial_velocity', 'l' and 'b'. Optionally, any of the
            photometric bands from Gaia can be set (as 'g_mag', 'bp_mag',
            'rp_mag') and any of the color indices can be set (as 'bp_rp',
            'bp_g' or 'g_rp').

        Returns
        -------
        expected:
            'numpy.ndarray'. When the instance is called, it returns a
            'np.ndarray' containing the expected values, in order, for 'l',
            'b', 'parallax', 'pml', 'pmb', 'radial_velocity' and, the
            photometric band and color index used.

        """
        self.__from_file = from_file

        if self.__from_file is True:
            self.__initialize_from_file(expected)
            return

        for k in keys:
            try:
                setattr(self, k, expected[k])
            except KeyError:
                raise KeyError(
                    f"Must include an expected value for {k}."
                    f"If unknown, use `None`.")

        for m in mags[:3]:
            if m in expected.keys():
                self.__mag = expected[m]
            else:
                self.__mag = np.nan
        for c in mags[3:]:
            if c in expected.keys():
                self.__c_index = expected[c]
            else:
                self.__c_index = np.nan

        ra_rad = np.deg2rad(self.ra)
        dec_rad = np.deg2rad(self.dec)
        l_rad = np.deg2rad(self.l)
        b_rad = np.deg2rad(self.b)

        p_icrs = np.array([-np.sin(ra_rad),
                           np.cos(ra_rad),
                           0])
        q_icrs = np.array([-np.cos(ra_rad) * np.sin(dec_rad),
                           -np.sin(ra_rad) * np.sin(dec_rad),
                           np.cos(dec_rad)])
        p_gal = np.array([-np.sin(l_rad),
                          np.cos(l_rad),
                          0])
        q_gal = np.array([-np.cos(l_rad) * np.sin(b_rad),
                          -np.sin(l_rad) * np.sin(b_rad),
                          np.cos(b_rad)])

        mu_icrs = p_icrs * self.pmra + q_icrs * self.pmdec
        mu_gal = A_G_INV.dot(mu_icrs)

        self.pml = np.dot(p_gal, mu_gal)
        self.pmb = np.dot(q_gal, mu_gal)

    def __initialize_from_file(self, expected):
        for k in data:
            setattr(self, k, expected[k][0])

        setattr(self, '_ExpectedValues__mag', expected['mag'][0])
        setattr(self, '_ExpectedValues__c_index', expected['c_index'][0])

    def save(self, path):
        array = self.__call__()
        save_array = np.zeros(1, dtype=save_dtype)
        for _, name in enumerate(save_dtype.names):
            save_array[name] = array[_]

        np.savetxt(
            path, save_array
        )

    @classmethod
    def load(cls, path):
        expected = np.genfromtxt(path, dtype=save_dtype).reshape((1,))

        return ExpectedValues(expected=expected, from_file=True)

    def __call__(self):
        arr = []
        for name in data:
            arr.append(getattr(self, name))
        arr.append(self.__mag)
        arr.append(self.__c_index)

        return arr
