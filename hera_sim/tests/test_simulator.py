"""
This tests the Simulator object and associated utilities. It does *not*
check for correctness of individual models, as they should be tested
elsewhere.
"""

import shutil
import tempfile
from os import path
from pyuvdata import UVData
import copy

import numpy as np
from nose.tools import raises, assert_raises

from hera_sim.foregrounds import diffuse_foreground
from hera_sim.noise import thermal_noise, HERA_Tsky_mdl
from hera_sim.simulate import Simulator, run_simulate
from hera_sim.data import DATA_PATH


def create_sim(autos=False):
    return Simulator(
        n_freq=10,
        n_times=20,
        antennas={
            0: (20.0, 20.0, 0),
            1: (50.0, 50.0, 0)
        },
        antpairs=None if autos else "cross"
    )


@raises(ValueError)
def test_wrong_antpairs():
    Simulator(
        n_freq=10,
        n_times=20,
        antennas={
            0: (20.0, 20.0, 0),
            1: (50.0, 50.0, 0)
        },
        antpairs="bad_specifier"
    )


@raises(KeyError)
def test_bad_antpairs():
    Simulator(
        n_freq=10,
        n_times=20,
        antennas={
            0: (20.0, 20.0, 0),
            1: (50.0, 50.0, 0)
        },
        antpairs=[(2, 2)]
    )


def test_from_empty():
    sim = create_sim()

    assert sim.data.data_array.shape == (20, 1, 10, 1)
    assert np.all(np.isclose(sim.data.data_array, 0))


def test_add_with_str():
    sim = create_sim()
    sim.add_eor("noiselike_eor")
    assert not np.all(np.isclose(sim.data.data_array, 0))


def test_add_with_builtin():
    sim = create_sim()
    sim.add_foregrounds(diffuse_foreground, Tsky_mdl=HERA_Tsky_mdl['xx'])
    assert not np.all(np.isclose(sim.data.data_array, 0))


def test_add_with_custom():
    sim = create_sim()

    def custom_noise(**kwargs):
        vis = thermal_noise(**kwargs)
        return 2 * vis

    sim.add_noise(custom_noise)
    assert not np.all(np.isclose(sim.data.data_array, 0))


def test_io():
    sim = create_sim()

    # Create a temporary directory to write stuff to (for python 3 this is much easier)
    direc = tempfile.mkdtemp()

    sim.add_foregrounds("pntsrc_foreground")
    sim.add_gains()

    sim.write_data(path.join(direc, 'tmp_data.uvh5'))

    sim2 = Simulator(
        data_filename=path.join(direc, 'tmp_data.uvh5')
    )

    assert np.all(sim.data.data_array == sim2.data.data_array)

    with assert_raises(ValueError):
        sim.write_data(path.join(direc, 'tmp_data.bad_extension'), file_type="bad_type")

    # delete the tmp
    shutil.rmtree(direc)


@raises(AttributeError)
def test_wrong_func():
    sim = create_sim()

    sim.add_eor("noiselike_EOR")  # wrong function name


@raises(TypeError)
def test_wrong_arguments():
    sim = create_sim()
    sim.add_foregrounds(diffuse_foreground, what=HERA_Tsky_mdl['xx'])


def test_other_components():
    sim = create_sim(autos=True)

    sim.add_rfi("rfi_stations")

    assert np.all(np.isclose(sim.data.data_array,  0))

    sim.add_xtalk('gen_whitenoise_xtalk', bls=[(0, 1, 'xx')])
    sim.add_xtalk('gen_cross_coupling_xtalk', bls=[(0, 1, 'xx')])
    sim.add_sigchain_reflections(ants=[0])

    assert not np.all(np.isclose(sim.data.data_array,  0))
    assert np.all(np.isclose(sim.data.get_data(0, 0),  0))


def test_not_add_vis():
    sim = create_sim()
    vis = sim.add_eor("noiselike_eor", add_vis=False)

    assert np.all(np.isclose(sim.data.data_array,  0))

    assert not np.all(np.isclose(vis, 0))

    assert "noiselike_eor" not in sim.data.history


def test_adding_vis_but_also_returning():
    sim = create_sim()
    vis = sim.add_eor("noiselike_eor", ret_vis=True)

    assert not np.all(np.isclose(vis, 0))
    np.testing.assert_array_almost_equal(vis, sim.data.data_array)

    vis = sim.add_foregrounds("diffuse_foreground", Tsky_mdl=HERA_Tsky_mdl['xx'], ret_vis=True)
    np.testing.assert_array_almost_equal(vis, sim.data.data_array)


def test_run_simulate():
    # load data
    filepath = path.join(DATA_PATH, "PyGSM_Jy_downselect.uvh5")
    uv = UVData()
    uv.read(filepath)
    bl = (23, 24, 'xx')

    # test add uvd
    uvd = run_simulate(uv, add_uvd=uv, add_amp=2.0, inplace=False)
    np.testing.assert_array_almost_equal(uv.get_data(bl) * 3.0, uvd.get_data(bl))

    # test add noise
    uvd = run_simulate(uv, add_noise=True, noise_amp=1.0, Trx=100.0, inplace=False, seed=0)
    assert not np.any(np.isclose(uvd.get_data(bl) - uv.get_data(bl), 0.0)) # assert data is different

    # test add xtalk
    uvd = run_simulate(uv, add_xtalk=True, xtalk_amp=1e-3, xtalk_dly=500., xtalk_phs=0., seed=0, inplace=False)
    assert not np.any(np.isclose(uvd.get_data(bl) - uv.get_data(bl), 0.0)) # assert data is different

    # test add reflection
    uvd = run_simulate(uv, add_ref=True, ref_amp=1e-3, ref_dly=500., ref_phs=0., seed=0, inplace=False)
    assert not np.any(np.isclose(uvd.get_data(bl) - uv.get_data(bl), 0.0)) # assert data is different

