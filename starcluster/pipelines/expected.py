import optparse as op
import os

from pathlib import Path

import astropy.units as u
import numpy as np

from astropy.coordinates import SkyCoord, Distance
from astropy.coordinates import Longitude, Latitude
from astroquery.simbad import Simbad

from starcluster.utils import ExpectedValues


def main():
    parser = op.OptionParser()

    parser.add_option('-n', '--name', dest='name', type='string',
                      help="Identifier of the object to query.")
    parser.add_option('-o', '--output-folder', dest='output', type='string',
                      default='./expected',
                      help="Output folder to save the expected values in.")
    parser.add_option('-m', '--magnitude', dest='mag', type='float',
                      default=np.nan,
                      help="The magnitude of the object to be shown. Default "
                           "is Nan.")
    parser.add_option('-c', '--color_index', dest='c_index', type='float',
                      default=np.nan,
                      help="The color index of the object to be shown. Default "
                           "is Nan.")
    (options, args) = parser.parse_args()

    obj_name = options.name.lower()

    out_folder = Path(options.output)
    if not out_folder.exists():
        os.mkdir(out_folder)

    out_path = out_folder.joinpath(f'{obj_name}_exp.csv')

    cs_simbad = Simbad()
    cs_simbad.add_votable_fields('parallax', 'pmra', 'pmdec', 'velocity')

    result = cs_simbad.query_object(obj_name)
    print(result.colnames)
    result = result.as_array()
    print(result)

    ra_list = un_string_coords(result['RA'][0])
    dec_list = un_string_coords(result['DEC'][0])
    ra = ra_list[0] + ':' + ra_list[1] + ':' + ra_list[2]
    dec = dec_list[0] + ':' + dec_list[1]
    ra = Longitude(ra, unit=u.hourangle)
    dec = Latitude(dec, unit=u.deg)

    parallax = result['PLX_VALUE'][0]

    sc_object = SkyCoord(
        ra=ra, dec=dec,
        radial_velocity=result['RVZ_RADVEL'][0]*u.km/u.s,
        pm_ra_cosdec=result['PMRA'][0]*u.mas/u.yr,
        pm_dec=result['PMDEC'][0]*u.mas/u.yr,
        distance=Distance(parallax=parallax*u.mas),
        frame='icrs',
    )

    expected_values = {
        'ra': sc_object.ra.deg,
        'dec': sc_object.dec.deg,
        'l': sc_object.galactic.l.deg,
        'b': sc_object.galactic.b.deg,
        'pmra': sc_object.pm_ra_cosdec.value,
        'pmdec': sc_object.pm_dec.value,
        'radial_velocity': sc_object.radial_velocity.value,
        'parallax': sc_object.distance.parallax.to(u.mas).value
    }

    expected_values = ExpectedValues(expected_values)

    expected_values.save(out_path)


def un_string_coords(str_):
    str_ = str_.split(' ')
    str_ = [s.split('+')[-1] for s in str_]

    for idx, s in enumerate(str_):
        if s.find('0') == 0 and len(s) > 1:
            str_[idx] = s[1:]

    return str_


if __name__ == "__main__":
    main()
