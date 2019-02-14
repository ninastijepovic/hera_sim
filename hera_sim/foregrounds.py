"""
A module with functions for generating foregrounds signals.

Each function may take arbitrary parameters, but should return a 2D array of visibilities for the requested baseline
at the requested lsts and frequencies.
"""

import aipy
import numpy as np
from scipy.interpolate import RectBivariateSpline

from . import noise
from . import utils


def diffuse_foreground(Tsky, lsts, fqs, bl_len_ns, bm_poly=noise.HERA_BEAM_POLY, scalar=30.,
                       fr_width=None, fr_max_mult=2.0):

    """
    Produce visibilities containing diffuse foregrounds.

    Args:
        Tsky (float): sky temperature [in mK?]
        lsts (ndarray): LSTs [radians]
        fqs (ndarray): frequencies [GHz]
        bl_len_ns (float): East-West baseline length [nanosec]
        bm_poly (ndarray): a polynomial for beam size with frequency.
        scalar (float): WHAT'S THIS?
        fr_width (float): width of Gaussian FR filter in 1 / sec
        fr_max_mult (float): multiplier of fr_max to get lst_grid resolution

    Returns:
        2D ndarray : visibilities at each lst, fq pair.
    """
    # If an auto-correlation, return the beam-weighted integrated sky.
    if utils.get_bl_len_magnitude(bl_len_ns) == 0:
        return Tsky(lsts, fqs) / noise.jy2T(fqs, bm_poly=bm_poly)

    # Get the maximum fringe rate corresponding to a time scale over
    # which co-ordinates pass through the beam.
    beam_widths = np.polyval(bm_poly, fqs)
    fr_max_beam = np.max(2*np.pi/(aipy.const.sidereal_day * beam_widths))
    fr_max = np.max(utils.calc_max_fringe_rate(fqs, bl_len_ns))

    fr_max = max(fr_max, fr_max_beam)

    dt = 1.0 / (fr_max_mult * fr_max)  # over-resolve by fr_mult factor
    ntimes = int(np.around(aipy.const.sidereal_day / dt))

    lst_grid = np.linspace(0, 2 * np.pi, ntimes, endpoint=False)
    nos = Tsky(lst_grid, fqs) * noise.white_noise((ntimes, fqs.size))

    nos, ff, frs = utils.rough_fringe_filter(nos, lst_grid, fqs, bl_len_ns,
                                             fr_width=fr_width, normalise=1)

    nos = utils.rough_delay_filter(nos, fqs, bl_len_ns, normalise=1)
    nos /= noise.jy2T(fqs, bm_poly=bm_poly)

    mdl_real = RectBivariateSpline(lst_grid, fqs, scalar * nos.real)
    mdl_imag = RectBivariateSpline(lst_grid, fqs, scalar * nos.imag)
    return mdl_real(lsts, fqs) + 1j * mdl_imag(lsts, fqs)


def pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=1000, Smin=0.3, Smax=300,
                      beta=-1.5, spectral_index_mean=-1, spectral_index_std=0.5,
                      reference_freq=0.15):
    """
    Generate visibilities from randomly placed point sources.

    Point sources drawn from a power-law source count distribution from 0.3 to 300 Jy, with index -1.5

    Args:
        lsts (ndarray): LSTs [radians]
        fqs (ndarray): frequencies [GHz]
        bl_len_ns (float): East-West baseline length [nanosec]
        nsrcs (int): number of sources to place in the sky

    Returns:
        2D ndarray : visibilities at each lst, fq pair.
    """
    ras = np.random.uniform(0, 2 * np.pi, nsrcs)
    indices = np.random.normal(spectral_index_mean, spectral_index_std, size=nsrcs)
    mfreq = reference_freq
    beam_width = (40 * 60.) * (mfreq / fqs) / aipy.const.sidereal_day * 2 * np.pi  # XXX hardcoded HERA

    # Draw flux densities from a power law between Smin and Smax with a slope of beta.
    flux_densities = ((Smax ** (beta + 1) - Smin ** (beta + 1)) * np.random.uniform(size=nsrcs) + Smin ** (
                beta + 1)) ** (1. / (beta + 1))

    vis = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    for ra, flux, index in zip(ras, flux_densities, indices):
        t = np.argmin(np.abs(utils.compute_ha(lsts, ra)))
        dtau = np.random.uniform(-.1 * bl_len_ns, .1 * bl_len_ns)  # XXX adds a bit to total delay, increasing bl_len_ns
        vis[t, :] += flux * (fqs / mfreq) ** index * np.exp(2j * np.pi * fqs * dtau)
    ha = utils.compute_ha(lsts, 0)
    for fi in xrange(fqs.size):
        bm = np.exp(-ha ** 2 / (2 * beam_width[fi] ** 2))
        bm = np.where(np.abs(ha) > np.pi / 2, 0, bm)
        w = .9 * bl_len_ns * np.sin(ha) * fqs[fi]  # XXX .9 to offset increase from dtau above

        phs = np.exp(2j * np.pi * w)
        kernel = bm * phs
        vis[:, fi] = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(vis[:, fi]))
    return vis
