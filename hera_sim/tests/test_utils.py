import numpy as np

from hera_sim import utils, noise


def test_bl_vec():
    bl = 1

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = (0, 1)

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = [0, 1]

    assert len(utils._get_bl_len_vec(bl)) == 3

    bl = np.array([0, 1, 2])

    assert len(utils._get_bl_len_vec(bl)) == 3


def test_delay_filter_norm():
    N = 50
    fqs = np.linspace(0.1, 0.2, N)

    tsky = np.ones(N)

    out = 0
    for i in range(1000):
        _noise = tsky * noise.white_noise(N)
        outnoise = utils.rough_delay_filter(_noise, fqs, 30, normalise=1)

        out += np.sum(np.abs(outnoise)**2)

    out /= 1000

    print(out, np.sum(tsky**2))
    assert np.isclose(out, np.sum(tsky**2), rtol=1e-2)