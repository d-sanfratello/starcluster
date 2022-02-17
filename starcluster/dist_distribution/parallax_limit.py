import numpy as np

from .const import R_GAL, R_S_SAGA


class ParallaxLimit:
    def __call__(self, l):
        """

        Parameters
        ----------
        l:
            'number'. The galactic longitude of the direction of observation,
            expressed in radians.

        Returns
        -------
        L(l):
            'float'. The upper distance limit of the distance from the edge
            of the Galaxy in the direction of galactic longitude l, expressed in
            kpc.
        """
        # FIXME: Complete docstring
        r_G2 = R_S_SAGA * R_S_SAGA
        R_G2 = R_GAL * R_GAL

        cos_term = l + np.arcsin(np.sin(l) * R_S_SAGA / R_GAL)
        cos_term = np.cos(cos_term)
        cos_term *= 2 * R_GAL * R_S_SAGA

        return np.sqrt(r_G2 + R_G2 + cos_term)

parallax_limit = ParallaxLimit()
