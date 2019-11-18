"""
Provides a number of mappings which may be useful for visibility simulators.
"""
import healpy
import numpy as np

def uvbeam_to_lm(uvbeam, freqs, n_pix_lm=63, trunc_at_horizon=False, **kwargs):
    """
    Convert a UVbeam to a uniform (l,m) grid

    Args:
        uvbeam (UVBeam): a UVBeam object
        freqs (1D array, shape=(NFREQS,)): frequencies to interpolate to [Hz]
        n_pix_lm (int, optional): number of pixels on a side for the beam grid.

    Returns:
        ndarray, shape[nfreq, beam_px, beam_px]: the beam map cube.
    """
    
    l = np.linspace(-1, 1, n_pix_lm, dtype=np.float32)
    l, m = np.meshgrid(l, l)
    l = l.flatten()
    m = m.flatten()

    lsqr = l ** 2 + m ** 2
    n = np.where(lsqr < 1, np.sqrt(1 - lsqr), 0)

    az = -np.arctan2(m, l)
    za = np.pi/2 - np.arcsin(n)
    
    efield_beam = uvbeam.interp(az, za, freqs, **kwargs)[0]
    efieldXX = efield_beam[0,0,1]
    
    # Get the relevant indices of res
    bm = np.zeros((len(freqs), len(l)))
    
    if trunc_at_horizon:
        bm[:, n >= 0] = efieldXX[:, n >= 0]
    else:
        bm = efieldXX
    
    if np.max(bm) > 0:
        bm /= np.max(bm)

    return bm.reshape((len(freqs), n_pix_lm, n_pix_lm))


def eq2top_m(ha, dec):
    """
    Return the 3x3 matrix converting equatorial coordinates to topocentric
    at the given hour angle (ha) and declination (dec).

    Ripped straight from aipy.
    """
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    zero = np.zeros_like(ha)

    map = np.array([[sin_H, cos_H, zero],
                    [-sin_d * cos_H, sin_d * sin_H, cos_d],
                    [cos_d * cos_H, -cos_d * sin_H, sin_d]])

    if len(map.shape) == 3:
        map = map.transpose([2, 0, 1])

    return map


def healpix_to_crd_eq(h, nest=False):
    """
    Determine equatorial co-ordinates of a healpix map's pixels.

    Args:
        h (1D array): the healpix array (must have size 12*N^2 for some N).
        nest (bool, optional): whether the healpix array is in NEST configuration.

    Returns:
        2D array, shape[12*N^2, 3]: the equatorial co-ordinates of each pixel.
    """
    assert h.ndim == 1, "h must be a 1D array"

    px = np.arange(len(h))
    crd_eq = np.array(healpy.pix2vec(healpy.get_nside(h), px, nest=nest), dtype=np.float32)
    return crd_eq
