import numpy as np
from pathlib import Path

from starcluster.extract_data import Data


class TestData:
    def test_read_cartesian(self):
        cart_path = Path('./cart_data.txt')

        dtype = np.dtype([('x', float),
                          ('y', float),
                          ('z', float),
                          ('vx', float),
                          ('vy', float),
                          ('vz', float)])
        fake_data = np.zeros(2, dtype=dtype)
        fake_data['x'] = ['1.', '2.']
        fake_data['y'] = ['3.', '4.']
        fake_data['z'] = ['5.', '6.']
        fake_data['vx'] = ['10.', '20.']
        fake_data['vy'] = ['30.', '40.']
        fake_data['vz'] = ['50.', '60.']
        np.savetxt(cart_path, fake_data,
                   header='x,y,z,vx,vy,vz',
                   delimiter=',')

        cart_data = Data(cart_path, is_cartesian=True)
        cart_saved_data = cart_data.read()

        assert type(cart_saved_data) == np.ndarray, \
            "Wrong type of recovered dataset."
        assert cart_saved_data.shape == fake_data.shape, \
            "Wrong shape of recovered data."
        assert np.array_equal(cart_saved_data, fake_data) is True, \
            "Invalid value in recovered structured array."

    def test_convert_from_gaia(self):
        gaia_path = Path('./gaia.csv')
        gaia_cart_path = Path('./gaia_galactic.txt')

        gaia_data = Data(gaia_path, is_cartesian=False)
        gaia_data.read(ruwe=1.4)

        gaia_cart = Data(gaia_cart_path, is_cartesian=True)
        gaia_rec = gaia_cart.read()

        rec_col = gaia_rec['x']
        assert len(rec_col) == 3,\
            "Something went wrong with the ruwe condition."

    def test_missing_data(self):
        gaia_path = Path('./missing.csv')
        gaia_cart_path = Path('./gaia_missing.txt')

        gaia_data = Data(gaia_path, is_cartesian=False)
        gaia_data.read(outpath='./gaia_missing.txt')

        gaia_cart = Data(gaia_cart_path, is_cartesian=True)
        gaia_rec = gaia_cart.read()

        for col in gaia_rec.dtype.names:
            assert np.isnan(gaia_rec[col]).all() == False, \
                "Soemthing went wrong with the NaN check. NaNs remaining."
        rec_col = gaia_rec['x']
        assert len(rec_col) == 3, \
            "Something went wrong with the NaN check. Too long record."
