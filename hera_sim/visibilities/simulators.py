import warnings

import healpy
import numpy as np
from cached_property import cached_property
from scipy.interpolate import RectBivariateSpline

from . import conversions


class VisibilitySimulator(object):
    _point_source_ability = True
    _diffuse_ability = True

    def __init__(self, freq, antpos, latitude, lsts, beams=None, beam_ids=None,
                 sky_intensity=None, point_sources=None, nside=50):
        """
        Base VisibilitySimulator class.

        Any actual visibility simulator should be sub-classed from this one.
        This class provides several convenience methods and defines the API.

        Note that for reasons of memory, the simulator deals with frequencies
        independently, and so only one frequency is passed. This allows you to
        load/unload data for different frequencies between calls.

        Args:
            freq (float):
                instrumental frequency [GHz]
            antpos (2D float array, shape=[N_ANTS, 3]):
                Antenna positions in ENU co-ordinates [metres]
            latitude (float):
                Latitude of centre of antenna array [rad].
            beams (2D array, optional, shape=[N_BEAMS, N_PIX_BEAM]:
                Healpix beam models for as many antennae as have unique beams.
                The array can be empty in order to not apply any beams.
            beam_ids (1D int array, optional, shape=[N_ANTS]):
                List of integers specifying which beam model each antenna uses
                (i.e. the index of `beams` which it should refer to). Negative
                values indicate absence of beam.
            lsts (1D array, shape=[N_LSTS]):
                The LSTs at which to generate visibilities.
            sky_intensity (1D array, shape=[N_PIX_SKY]):
                A healpix model for the intensity of the sky emission.
            point_sources (2D array, optional, shape=[N_SOURCES, 3]):
                An array of point sources. For each source, the entries are
                (ra, dec, flux_density [Jy]).
            nside (int, optional):
                Only used if sky_intensity is *not* given but the simulator
                is incapable of directly dealing with point sources. In this
                case, it sets the resolution of the healpix map to which the
                sources will be allocated.
        """
        self.freq = freq
        self.antpos = antpos
        self.beams = np.array([]) if beams is None else beams
        self.lsts = lsts
        self.sky_intensity = sky_intensity
        self.beam_ids = -1 * np.ones(self.n_ant, dtype=np.int) if beam_ids is None else beam_ids
        self.latitude = latitude
        self._nside = nside

        self.point_sources = point_sources

        self.validate()

    def validate(self):
        if self.sky_intensity is not None and not healpy.isnpixok(self.n_pix):
            raise ValueError("The sky_intensity map is not compatible with healpy")

        if len(self.beams) and not healpy.isnpixok(self.beams.shape[1]):
            raise ValueError("The beam maps are not compatible with healpy")

        if self.point_sources is None and self.sky_intensity is None:
            raise ValueError("You must pass at least one of sky_intensity or "
                             "point_sources.")

        if self.antpos.shape[1] != 3:
            raise ValueError("The number of co-ordinate dimensions for antennae "
                             "should be three.")

        if np.max(self.beam_ids) >= self.n_beams:
            raise ValueError("The number of beams provided must be at least as "
                             "great as the greatest beam_id")

        if self.sky_intensity is not None and self.sky_intensity.ndim != 1:
            raise ValueError("sky_intensity must be a 1D array (a healpix map)")

        if not self._point_source_ability and self.point_sources is not None:
            warnings.warn("This visibility simulator is unable to explicitly "
                          "simulate point sources. Adding point sources to "
                          "diffuse pixels")
            if self.sky_intensity is None: self.sky_intensity = np.zeros(healpy.nside2npix(self.nside))
            self.sky_intensity += self.convert_point_sources_to_healpix(
                self.point_sources, self.nside
            )

        if not self._diffuse_ability and self.sky_intensity is not None:
            warnings.warn("This visibility simulator is unable to explicitly "
                          "simulate diffuse structure. Converting diffuse "
                          "intensity to approximate points")
            if self.point_sources is None: self.point_sources = 0
            self.point_sources += self.convert_healpix_to_point_sources(self.sky_intensity)

    @staticmethod
    def convert_point_sources_to_healpix(point_sources, nside=40):
        """
        Convert a set of point sources to an approximate diffuse healpix model.

        The healpix map returned is in RING scheme.

        Returns:
            1D array: the healpix diffuse model.
        """

        hmap = np.zeros(healpy.nside2npix(nside))

        # Get which pixel every point source lies in.
        pix = healpy.ang2pix(nside, point_sources[:, 0], point_sources[:, 1])

        hmap[pix] += point_sources[:, 2] / healpy.nside2pixarea(nside)

        return hmap

    @staticmethod
    def convert_healpix_to_point_sources(hmap):
        """
        Convert a healpix map to a set of point sources located at the centre
        of each pixel.

        Args:
            hmap (1D array):
                The healpix map.
        Returns:
            2D array: the point sources
        """
        nside = healpy.get_nside(hmap)
        ra, dec = healpy.pix2ang(nside, np.arange(len(hmap)))
        flux = hmap * healpy.nside2pixarea(nside)
        return np.array([ra, dec, flux])

    def simulate(self):
        pass

    @property
    def nside(self):
        if self.sky_intensity is not None:
            return healpy.get_nside(self.sky_intensity)
        else:
            return self._nside

    @cached_property
    def n_ant(self):
        """Number of antennas in array"""
        return self.antpos.shape[0]

    @cached_property
    def n_lsts(self):
        """Number of times (LSTs)"""
        return self.lsts.shape[0]

    @cached_property
    def n_beams(self):
        """Number of beam models used."""
        return self.beams.shape[0]

    @cached_property
    def n_pix(self):
        """Number of pixels in the sky map"""
        return self.sky_intensity.size

    @cached_property
    def n_pix_beam(self):
        """Number of pixels in the beam maps"""
        try:
            return self.beams.shape[1]
        except IndexError:
            return 0


class VisCPU(VisibilitySimulator):
    _point_source_ability = False

    def __init__(self, bm_pix=100, real_dtype=np.float32, complex_dtype=np.complex64, **kwargs):
        """
        Fast visibility simulator on the CPU.

        Args:
            bm_pix (int, optional): the number of pixels along a side in the
                beam map when converted to (l,m).
            real_dtype: a valid numpy dtype
            complex_dtype: a valid numpy dtype
            **kwargs:
                All arguments of :class:`VisibilitySimulator`.

        """
        self._real_dtype = real_dtype
        self._complex_dtype = complex_dtype
        self.bm_pix = bm_pix

        super(VisCPU, self).__init__(**kwargs)

        # Convert some of our arguments to forms more simple for vis_cpu
        self.antpos = self.antpos.astype(self._real_dtype)

    def get_beam_lm(self):
        """
        Obtain the beam pattern in (l,m) co-ordinates for each beam.

        Returns:
              3D array, shape[NANT, BM_PIX, BM_PIX]: the beam pattern in (l,m)
                  for each antenna.

        Note:
            Due to using the verbatim :func:`vis_cpu` function, the beam cube
            must have an entry for each antenna, which is a bit of a waste of
            memory in some cases. If this is changed in the future, this
            method can be modified to only return one matrix for each beam.

        """
        return np.array([
            conversions.beam_healpix_to_lm(
                self.beams[self.beam_ids[i]], self.bm_pix
            ) if self.beam_ids[i] >= 0 else np.ones((self.bm_pix, self.bm_pix))
            for i in range(self.n_ant)
        ])

    def get_crd_eq(self):
        """Calculate the equatorial co-ordinates of the healpix sky pixels."""
        return conversions.healpix_to_crd_eq(self.sky_intensity).astype(self._real_dtype)

    def get_eq2tops(self):
        """
        Calculate the set of 3x3 transformation matrices converting equatorial
        coords to topocentric at each LST.
        """
        return conversions.eq2top_m(self.lsts.astype(self._real_dtype),
                                    (self.latitude * np.ones_like(self.lsts)).astype(self._real_dtype)).astype(
            self._real_dtype)

    def simulate(self):
        """
        Runs the cpu_vis algorithm.

        Returns:
            array_like, shape(NTIMES, NANTS, NANTS): visibilities

        Notes:
            This routine does not support negative intensity values on the sky.
        """
        return vis_cpu(
            antpos=self.antpos.astype(self._real_dtype),
            freq=self.freq,
            eq2tops=self.get_eq2tops(),
            crd_eq=self.get_crd_eq(),
            I_sky=self.sky_intensity,
            bm_cube=self.get_beam_lm(),
            real_dtype=self._real_dtype,
            complex_dtype=self._complex_dtype
        )


def vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube, real_dtype=np.float32, complex_dtype=np.complex64):
    """
    Calculate visibility from an input intensity map and beam model.

    Args:
        antpos (array_like, shape: (NANT, 3)): antenna position array.
        freq (float): frequency to evaluate the visibilities at [GHz].
        eq2tops (array_like, shape: (NTIMES, 3, 3)): Set of 3x3 transformation matrices converting equatorial
            coordinates to topocentric at each hour angle (and declination) in the dataset.
        crd_eq (array_like, shape: (3, NPIX)): equatorial coordinates of Healpix pixels.
        I_sky (array_like, shape: (NPIX,)): intensity distribution on the sky, stored as array of Healpix pixels.
        bm_cube (array_like, shape: (NANT, BM_PIX, BM_PIX)): beam maps for each antenna.
        real_dtype, complex_dtype (dtype, optional): data type to use for real and complex-valued arrays.

    Returns:
        array_like, shape(NTIMES, NANTS, NANTS): visibilities
    """
    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)"
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)"
    ncrd, npix = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NPIX)"
    assert I_sky.ndim == 1 and I_sky.shape[0] == npix, "I_sky must have shape (NPIX,)"
    bm_pix = bm_cube.shape[-1]
    assert bm_cube.shape == (
        nant,
        bm_pix,
        bm_pix,
    ), "bm_cube must have shape (NANTS, BM_PIX, BM_PIX)"

    # Intensity distribution (sqrt) and antenna positions
    Isqrt = np.sqrt(I_sky).astype(real_dtype)  # XXX does not support negative sky
    antpos = antpos.astype(real_dtype)
    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages
    A_s = np.empty((nant, npix), dtype=real_dtype)
    vis = np.empty((ntimes, nant, nant), dtype=complex_dtype)
    tau = np.empty((nant, npix), dtype=real_dtype)
    v = np.empty((nant, npix), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    bm_pix_x = np.linspace(-1, 1, bm_pix)
    bm_pix_y = np.linspace(-1, 1, bm_pix)

    # Loop over time samples
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
        for i in range(nant):
            # Linear interpolation of primary beam pattern
            spline = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1, ky=1)
            A_s[i] = spline(ty, tx, grid=False)
        A_s = np.where(tz > 0, A_s, 0)

        # Calculate delays
        np.dot(antpos, crd_top, out=tau)
        np.exp((1.0j * ang_freq) * tau, out=v)

        # Complex voltages
        v *= A_s * Isqrt

        # Compute visibilities (upper triangle only)
        for i in range(len(antpos)):
            np.dot(v[i: i + 1].conj(), v[i:].T, out=vis[t, i: i + 1, i:])

    # Conjugate visibilities
    np.conj(vis, out=vis)

    # Fill in whole visibility matrix from upper triangle
    for i in range(nant):
        vis[:, i + 1:, i] = vis[:, i, i + 1:].conj()

    return vis
