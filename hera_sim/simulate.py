"""
Primary interface module for hera_sim, defining a :class:`Simulator` class which provides a common API for all
effects produced by this package.
"""

import functools
import inspect
import sys
import warnings
import copy

import numpy as np
from cached_property import cached_property
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from astropy import constants as const

from . import io
from . import sigchain
from . import noise
from .version import version


class CompatibilityException(ValueError):
    pass


def _get_model(mod, name):
    return getattr(sys.modules["hera_sim." + mod], name)


class _model(object):
    """
    A decorator that defines a "model" addition for the Simulator class.

    The basic functionality of the model is to:

    1. Provide keywords "add_vis" and "ret_vis" to enable adding the resulting
       visibilities to the underlying dataset or returning the added visibilities.
    2. Automatically locate a callable model provided either that callable
       or a string function name (the module from which the callable is imported
       can be passed to the decorator, but is by default intepreted as the last
       part of the model name).
    3. Add a comment to the `history` of the UVData object concerning what
       exactly has ben added.
    """

    def __init__(self, base_module=None, multiplicative=False):
        self.base_module = base_module
        self.multiplicative = multiplicative

    def __call__(self, func, *args, **kwargs):
        name = func.__name__

        @functools.wraps(func)
        def new_func(obj, *args, **kwargs):

            # If "ret_vis" is set, then we want to return the visibilities
            # that are being added to the base. If add_vis is set to False,
            # we need to
            add_vis = kwargs.pop("add_vis", True)

            ret_vis = kwargs.pop("ret_vis", False)
            if not add_vis:
                ret_vis = True

            if ret_vis:
                initial_vis = obj.data.data_array.copy()

            # If this is a multiplicative model, and *no* additive models
            # have been called, raise a warning.
            if self.multiplicative and np.all(obj.data.data_array == 0):
                warnings.warn("You are trying to determine visibilities that depend on preceding visibilities, but " +
                              "no previous vis have been created.")
            elif not self.multiplicative and (hasattr(obj, "_added_models") and any([x[1] for x in obj._added_models])):
                # some of the previous models were multiplicative, and now we're trying to add.
                warnings.warn("You are adding absolute visibilities _after_ determining visibilities that should " +
                              "depend on these. Please re-consider.")

            if "model" in inspect.getargspec(func)[0]:
                # Cases where there is a choice of model
                model = args[0] if args else kwargs.pop("model")

                # If the model is a str, get its actual callable.
                if isinstance(model, str):
                    if self.base_module is None:
                        self.base_module = name[4:]  # get the bit after the "add"

                    model = _get_model(self.base_module, model)

                func(obj, model, **kwargs)

                if not isinstance(model, str):
                    method = model.__name__
                else:
                    method = model

                method = "using {} ".format(method)
            else:
                # For cases where there is no choice of model.
                method = ""
                func(obj, *args, **kwargs)

            if add_vis:
                msg = "\nhera_sim v{version}: Added {component} {method_name}with kwargs: {kwargs}"
                obj.data.history += msg.format(
                    version=version,
                    component="".join(name.split("_")[1:]),
                    method_name=method,
                    kwargs=kwargs,
                )

                # Add this particular model to a cache of "added models" for this sim.
                # This can be gotten from history, but easier just to keep it here.
                if not hasattr(obj, "_added_models"):
                    obj._added_models = [(name, self.multiplicative)]
                else:
                    obj._added_models += [(name, self.multiplicative)]

            # Here actually return something.
            if ret_vis:
                res = obj.data.data_array - initial_vis

                # If we don't want to add the visibilities, set them back
                # to the original before returning.
                if not add_vis:
                    obj.data.data_array[:] = initial_vis[:]

                return res

        return new_func


class Simulator:
    """
    Primary interface object for hera_sim.

    Produces visibility simulations with various independent sky- and instrumental-effects, and offers the resulting
    visibilities in :class:`pyuvdata.UVData` format.
    """

    def __init__(
            self,
            data_filename=None,
            data = None,
            refresh_data=False,
            n_freq=None,
            n_times=None,
            antennas=None,
            **kwargs
    ):
        """
        Initialise the object either from file or by creating an empty object.

        Args:
            data_filename (str, optional): filename of data to be read, in ``pyuvdata``-compatible format. If not
                given, an empty :class:`pyuvdata.UVdata` object will be created from scratch. *Deprecated since
                v0.0.1, will be removed in v0.1.0. Use `data` instead*.
            data (str or :class:`UVData`): either a string pointing to data to be read (i.e. the same as
                `data_filename`), or a UVData object.
            refresh_data (bool, optional): if reading data from file, this can be used to manually set the data to zero,
                and remove flags. This is useful for using an existing file as a template, but not using its data.
            n_freq (int, optional): if `data_filename` not given, this is required and sets the number of frequency
                channels.
            n_times (int, optional): if `data_filename` is not given, this is required and sets the number of obs
                times.
            antennas (dict, optional): if `data_filename` not given, this is required. See docs of
                :func:`~io.empty_uvdata` for more details.

        Other Args:
            All other arguments are sent either to :func:`~UVData.read` (if `data_filename` is given) or
            :func:`~io.empty_uvdata` if not. These all have default values as defined in the documentation for those
            objects, and are therefore optional.

        Raises:
            :class:`CompatibilityException`: if the created/imported data has attributes which are in conflict
                with the assumptions made in the models of this Simulator.

        """

        if data_filename is not None:
            warnings.warn("`data_filename` is deprecated, please use `data` instead", DeprecationWarning)
            
        self.data_filename = data_filename

        if self.data_filename is None and data is None:
            # Create an empty UVData object.

            # Ensure required parameters have been set.
            if n_freq is None:
                raise ValueError("if data_filename and data not given, n_freq must be given")
            if n_times is None:
                raise ValueError("if data_filename and data not given, n_times must be given")
            if antennas is None:
                raise ValueError("if data_filename and data not given, antennas must be given")

            # Actually create it
            self.data = io.empty_uvdata(
                nfreq=n_freq,
                ntimes=n_times,
                ants=antennas,
                **kwargs
            )

        else:
            if type(data) == str:
                self.data_filename = data

            if self.data_filename is not None:
                # Read data from file.
                self.data = self._read_data(self.data_filename, **kwargs)

                # Reset data to zero if user desires.
                if refresh_data:
                    self.data.data_array[:] = 0.0
                    self.data.flag_array[:] = False
                    self.data.nsample_array[:] = 1.0
            elif self.data is not None:
                self.data = data

        # Check if the created/read data is compatible with the assumptions of
        # this class.
        self._check_compatibility()

    @cached_property
    def antpos(self):
        """
        Dictionary of {antenna: antenna_position} for all antennas in the data.
        """
        antpos, ants = self.data.get_ENU_antpos(pick_data_ants=True)
        return dict(zip(ants, antpos))

    @staticmethod
    def _read_data(filename, **kwargs):
        uv = UVData()
        uv.read(filename, read_data=True, **kwargs)
        return uv

    def write_data(self, filename, file_type="uvh5", **kwargs):
        """
        Write current UVData object to file.

        Args:
            filename (str): filename to write to.
            file_type: (str): one of "miriad", "uvfits" or "uvh5" (i.e. any of the supported write methods of
                :class:`pyuvdata.UVData`) which determines which write method to call.
            **kwargs: keyword arguments sent directly to the write method chosen.
        """
        try:
            getattr(self.data, "write_%s" % file_type)(filename, **kwargs)
        except AttributeError:
            raise ValueError("The file_type must correspond to a write method in UVData.")

    def _check_compatibility(self):
        """
        Merely checks the compatibility of the data with the assumptions of the simulator class and its modules.
        """
        if self.data.phase_type != "drift":
            raise CompatibilityException("The phase_type of the data must be 'drift'.")

    def _iterate_antpair_pols(self):
        """
        Iterate through antenna pairs and polarizations in the data object
        """

        for ant1, ant2, pol in self.data.get_antpairpols():
            blt_inds = self.data.antpair2ind((ant1, ant2))
            pol_ind = self.data.get_pols().index(pol)
            yield ant1, ant2, pol, blt_inds, pol_ind

    @_model()
    def add_eor(self, model, **kwargs):
        """
        Add an EoR-like model to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.eor`, or
                a callable which has the signature ``fnc(lsts, fqs, bl_vec, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the EoR model function, other than `lsts`, `fqs` and `bl_vec`.
        """
        # frequencies come from zeroths spectral window
        fqs = self.data.freq_array[0] * 1e-9

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            lsts = self.data.lst_array[blt_ind]
            bl_vec = (self.antpos[ant1] - self.antpos[ant2]) * 1e9 / const.c.value
            vis = model(lsts=lsts, fqs=fqs, bl_vec=bl_vec, **kwargs)
            self.data.data_array[blt_ind, 0, :, pol_ind] += vis

    @_model()
    def add_foregrounds(self, model, **kwargs):
        """
        Add a foreground model to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.foregrounds`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_vec, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the foregournd model function, other than `lsts`, `fqs` and `bl_vec`.
        """
        # frequencies come from zeroth spectral window
        fqs = self.data.freq_array[0] * 1e-9

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            lsts = self.data.lst_array[blt_ind]
            bl_vec = (self.antpos[ant1] - self.antpos[ant2]) * 1e9 / const.c.value
            vis = model(lsts, fqs, bl_vec, **kwargs)
            self.data.data_array[blt_ind, 0, :, pol_ind] += vis

    @_model()
    def add_noise(self, model, **kwargs):
        """
        Add thermal noise to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.noise`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, omega_p, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the noise model function, other than `lsts`, `fqs` and `bl_len_ns`.
        """
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            lsts = self.data.lst_array[blt_ind]

            self.data.data_array[blt_ind, 0, :, pol_ind] += model(
                lsts=lsts, fqs=self.data.freq_array[0] * 1e-9, **kwargs
            )

    @_model()
    def add_rfi(self, model, **kwargs):
        """
        Add RFI to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.rfi`,
                or a callable which has the signature ``fnc(lsts, fqs, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the RFI model function, other than `lsts` or `fqs`.
        """
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            lsts = self.data.lst_array[blt_ind]

            # RFI added in-place
            model(
                lsts=lsts,
                # Axis 0 is spectral windows, of which at this point there are always 1.
                fqs=self.data.freq_array[0] * 1e-9,
                rfi=self.data.data_array[blt_ind, 0, :, 0],
                **kwargs
            )

    @_model(multiplicative=True)
    def add_gains(self, **kwargs):
        """
        Apply mock gains to visibilities.

        Currently this consists of a bandpass, and cable delays & phases.

        Args:
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the gen_gains method in :mod:~`hera_sim.sigchain`.
        """

        gains = sigchain.gen_gains(
            fqs=self.data.freq_array[0] * 1e-9, ants=self.data.get_ants(), **kwargs
        )

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            self.data.data_array[blt_ind, 0, :, pol_ind] = sigchain.apply_gains(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                gains=gains,
                bl=(ant1, ant2)
            )

    @_model(multiplicative=True)
    def add_sigchain_reflections(self, ants=None, **kwargs):
        """
        Apply signal chain reflections to visibilities.

        Args:
            ants: list of antenna numbers to add reflections to
            **kwargs: keyword arguments sent to the gen_reflection_gains method in :mod:~`hera_sim.sigchain`.
        """
        if ants is None:
            ants = self.data.get_ants()
            
        # generate gains
        gains = sigchain.gen_reflection_gains(self.data.freq_array[0], ants, **kwargs)

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            self.data.data_array[blt_ind, 0, :, pol_ind] = sigchain.apply_gains(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                gains=gains,
                bl=(ant1, ant2)
            )

    @_model('sigchain', multiplicative=True)
    def add_xtalk(self, model='gen_whitenoise_xtalk', bls=None, **kwargs):
        """
        Add crosstalk to visibilities.

        Args:
            bls (list of 3-tuples, optional): ant-pair-pols to add xtalk to.
            **kwargs: keyword arguments sent to the model :meth:~`hera_sim.sigchain.{model}`.
        """
        freqs = self.data.freq_array[0]
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            if bls is not None and (ant1, ant2, pol) not in bls:
                continue
            if model.__name__ == 'gen_whitenoise_xtalk':
                xtalk = model(freqs, **kwargs)
            elif model.__name__ == 'gen_cross_coupling_xtalk':
                # for now uses ant1 ant1 for auto correlation vis
                autovis = self.data.get_data(ant1, ant1, pol)
                xtalk = model(freqs, autovis, **kwargs)

            self.data.data_array[blt_ind, 0, :, pol_ind] = sigchain.apply_xtalk(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                xtalk=xtalk
            )


def run_simulate(uvd, add_uvd=None, add_amp=1.0, bls=None, pols=None, filetype='uvh5',
                 add_noise=False, noise_amp=1.0, Trx=100.0, add_xtalk=False, xtalk_amp=None,
                 xtalk_dly=None, xtalk_phs=None, add_ref=False, ref_amp=None, ref_dly=None, ref_phs=None,
                 seed=0, run_check=True, inplace=True):
    """
    A user interface for utilizing the functionality in hera_sim.

    uvd --> add_noise --> add_xtalk --> add_ref --> output

    Args:
        uvd (str or UVData): A data object or filepath to simulate on top of.
        add_uvd (str or UVData): Data object to add to uvd.
        add_amp (float): Amplitude with which to multiply add_uvd data before summing with uvd.
        bls (list of tuples): Baseline selection for read-in if inputs are filepaths
        pols (list of str): Polarization selection for read-in if inputs are filepaths
        filetype (str): Filetype for read-in
        add_noise (bool): If True, add thermal noise to visibilities using auto-correlation in uvd.
        noise_amp (float): Multiplicative factor of noise before summing with uvd
        Trx (float): Reciever temperature in Kelvin (frequency independent)
        add_xtalk (bool): If True, add crosstalk (aka cross coupling or mutual coupling) to data
            using auto-correlation in uvd.
        xtalk_amp : (float or list of floats): amplitude with which to insert xtalk. A list of amps
            can be fed for multiple xtalk injections.
        xtalk_dly : (float or list of floats): delays [ns] to insert xtalk. A list of delays
            can be fed for multiple xtalk injections.
        xtalk_phs : (float or list of floats): Phase [radian] to insert xtalk. A list of phases
            can be fed for multiple xtalk injections.
        add_ref (bool): If True, add reflections to data.
        ref_amp (float or list of floats): Amplitude with which to insert reflection. A list
            can be fed for multiple reflections.
        ref_dly (float or list of floats): Delays [ns] to insert reflections. A list can be
            fed for multiple reflections.
        ref_phs (float or list of floats): Phases [radian] with which to insert reflections. A list can
            be fed for multiple reflections.
        seed (int): Random seed to use at the start of each component block.
        run_check (bool): If True, perform UVData check before return or write-out
        inplace (bool): If True, add components in place, otherwise make and return a deepcopy
    """
    # load uvd if necessary
    if isinstance(uvd, (str, np.str)):
        _uvd = UVData()
        _uvd.read(uvd, bls=bls, polarizations=pols)
        uvd = _uvd
    else:
        if not inplace:
            uvd = copy.deepcopy(uvd)

    # get metadata
    freqs = np.unique(uvd.freq_array) / 1e9
    lsts = []
    for l in uvd.lst_array:
        if l not in lsts:
            lsts.append(l)
    lsts = np.array(lsts)
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=False)
    Nants = len(ants)
    antpos_d = dict(zip(ants, antpos))
    times = np.unique(uvd.time_array)
    Ntimes = len(times)
    Nfreqs = len(freqs)
    antpairs = uvd.get_antpairpols()
    inttime = np.median(uvd.integration_time)

    # get baseline info
    bl_inds, pol_inds = {}, {}
    for ap in antpairs:
        # get baseline info
        pol = ap[-1]
        bl_ind = uvd.antpair2ind(ap, ordered=False)
        pol_ind = np.where(uvd.polarization_array == uvutils.polstr2num(pol))[0][0]
        bl_inds[ap] = bl_ind
        pol_inds[ap] = pol_ind

    # add uvd if necessary
    if add_uvd is not None:
        # if str, load them
        if isinstance(add_uvd, (str, np.str)):
            _add_uvd = UVData()
            _add_uvd.read(add_uvd, bls=bls, polarizations=pols)
            add_uvd = _add_uvd

        # iterate over antpairs
        _antpairs = add_uvd.get_antpairpols()
        for ap in _antpairs:
            if ap not in antpairs:
                continue
            uvd.data_array[bl_inds[ap], 0, :, pol_inds[ap]] += add_uvd.get_data(ap) * add_amp

    # get first auto-correlation if needed
    if add_noise or add_xtalk:
        autokeys = [k for k in antpairs if k[0] == k[1]]
        autocorr_Jy = uvd.get_data(autokeys[0])
        omega_p = noise.bm_poly_to_omega_p(freqs)
        autocorr_K = autocorr_Jy * noise.jy2T(freqs, omega_p) / 1e3

    # add noise
    if add_noise:
        # set seed
        np.random.seed(seed)

        # iterate over baselines
        for ap in antpairs:
            # construct Noise
            N = noise.sky_noise_jy(autocorr_K + Trx, freqs, lsts, omega_p, inttime=inttime) * noise_amp
            if ap[0] == ap[1]:
                N = np.abs(N)
            uvd.data_array[bl_inds[ap], 0, :, pol_inds[ap]] += N


    # add xtalk
    if add_xtalk:
        # set seed
        np.random.seed(seed)

        # parse xtalk parameters
        if isinstance(xtalk_amp, (float, np.float, int, np.int)):
            xtalk_amp = [xtalk_amp]
        if isinstance(xtalk_dly, (float, np.float, int, np.int)):
            xtalk_dly = [xtalk_dly]
        if isinstance(xtalk_phs, (float, np.float, int, np.int)):
            xtalk_phs = [xtalk_phs]

        # generate xtalk model
        X = np.zeros_like(autocorr_Jy, np.complex128)

        # iterate over xtalk injections
        for i in range(len(xtalk_dly)):
            X += sigchain.gen_cross_coupling_xtalk(freqs, autocorr_Jy, amp=xtalk_amp[i],
                                                   dly=xtalk_dly[i], phs=xtalk_phs[i])

        # iterate over antpairs
        for ap in antpairs:
            if ap[0] != ap[1]:
                uvd.data_array[bl_inds[ap], 0, :, pol_inds[ap]] += X

    # add reflections
    if add_ref:
        # set seed
        np.random.seed(seed)

        # parse reflection parameters
        if isinstance(ref_amp, (float, np.float, int, np.int)):
            ref_amp = [ref_amp]
        if isinstance(ref_dly, (float, np.float, int, np.int)):
            ref_dly = [ref_dly]
        if isinstance(ref_phs, (float, np.float, int, np.int)):
            ref_phs = [ref_phs]

        # build up gains
        gains = dict([(ant, np.ones((Ntimes, Nfreqs), np.complex128)) for ant in ants])
        for amp, dly, phs in zip(ref_amp, ref_dly, ref_phs):
            _amp = [amp] * Nants
            _dly = [dly] * Nants
            _phs = [phs] * Nants
            _gain = sigchain.gen_reflection_gains(freqs, ants, amp=_amp, dly=_dly, phs=_dly, conj=False)
            for key in _gain:
                if key in gains:
                    gains[key] *= _gain[key]

        # multiply into data
        for ap in antpairs:
            uvd.data_array[bl_inds[ap], 0, :, pol_inds[ap]] = sigchain.apply_gains(uvd.get_data(ap), gains, ap)

    if run_check:
        uvd.check()

    return uvd





