"""
Primary interface module for hera_sim, defining a :class:`Simulator` class which provides a common API for all
effects produced by this package.
"""

import functools
import inspect
import sys

from pyuvdata import UVData

from . import io
from . import sigchain
from .version import version


class CompatibilityException(ValueError):
    pass


def _get_model(mod, name):
    return getattr(sys.modules["hera_sim." + mod], name)


class _model(object):

    def __init__(self, base_module=None):
        self.base_module = base_module

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
                func(obj, **kwargs)

            if add_vis:
                msg = "\nhera_sim v{version}: Added {component} {method_name}with kwargs: {kwargs}"
                obj.data.history += msg.format(
                    version=version,
                    component="".join(name.split("_")[1:]),
                    method_name=method,
                    kwargs=kwargs,
                )

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
            refresh_data=False,
            n_freq=None,
            n_times=None,
            antennas=None,
            ant_pairs=None,
            **kwargs
    ):
        """
        Initialise the object either from file or by creating an empty object.

        Args:
            data_filename (str, optional): filename of data to be read, in ``pyuvdata``-compatible format. If not
                given, an empty :class:`pyuvdata.UVdata` object will be created from scratch.
            refresh_data (bool, optional): if reading data from file, this can be used to manually set the data to zero,
                and remove flags. This is useful for using an existing file as a template, but not using its data.
            n_freq (int, optional): if `data_filename` not given, this is required and sets the number of frequency
                channels.
            n_times (int, optional): if `data_filename` is not given, this is required and sets the number of obs
                times.
            antennas (dict, optional): if `data_filename` not given, this is required. See docs of
                :func:`~io.empty_uvdata` for more details.
            ant_pairs (list of 2-tuples, optional): if `data_filename` not given, this is required. See docs of
                :func:`~io.empty_uvdata` for more details.

        Other Args:
            All other arguments are sent either to :func:`~UVData.read` (if `data_filename` is given) or
            :func:`~io.empty_uvdata` if not. These all have default values as defined in the documentation for those
            objects, and are therefore optional.

        Raises:
            :class:`CompatibilityException`: if the created/imported data has attributes which are in conflict
                with the assumptions made in the models of this Simulator.

        """

        self.data_filename = data_filename

        if self.data_filename is None:
            # Create an empty UVData object.

            # Ensure required parameters have been set.
            assert (
                    n_freq is not None
            ), "if data_filename not given, n_freq must be given"
            assert (
                    n_times is not None
            ), "if data_filename not given, n_times must be given"
            assert (
                    antennas is not None
            ), "if data_filename not given, antennas must be given"
            assert (
                    ant_pairs is not None
            ), "if data_filename not given, ant_pairs must be given"

            # Actually create it
            self.data = io.empty_uvdata(
                nfreq=n_freq,
                ntimes=n_times,
                ants=antennas,
                antpairs=ant_pairs,
                **kwargs
            )

        else:
            # Read data from file.
            self.data = self._read_data(self.data_filename, **kwargs)

            # Reset data to zero if user desires.
            if refresh_data:
                self.data.data_array[:] = 0.0
                self.data.flag_array[:] = False
                self.data.nsample_array[:] = 1.0

        # Check if the created/read data is compatible with the assumptions of
        # this class.
        self._check_compatibility()

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
                a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the EoR model function, other than `lsts`, `fqs` and `bl_len_ns`.
        """

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            lsts = self.data.lst_array[blt_ind]
            bl_len_m = self.data.uvw_array[blt_ind][
                0, 0
            ]  # just the E-W baseline length at this point.

            self.data.data_array[blt_ind, 0, :, pol_ind] += model(
                lsts=lsts,
                # Axis 0 is spectral windows, of which at this point there are always 1.
                fqs=self.data.freq_array[0] * 1e-9,
                bl_len_ns=bl_len_m * 1e9 / 3e8,
                **kwargs
            )

    @_model()
    def add_foregrounds(self, model, **kwargs):
        """
        Add a foreground model to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.foregrounds`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the foregournd model function, other than `lsts`, `fqs` and `bl_len_ns`.
        """
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            lsts = self.data.lst_array[blt_ind]
            bl_len_m = self.data.uvw_array[blt_ind][
                0, 0
            ]  # just the E-W baseline length at this point.

            self.data.data_array[blt_ind, 0, :, pol_ind] += model(
                lsts=lsts,
                # Axis 0 is spectral windows, of which at this point there are always 1.
                fqs=self.data.freq_array[0] * 1e-9,
                bl_len_ns=bl_len_m * 1e9 / 3e8,
                **kwargs
            )

    @_model()
    def add_noise(self, model, **kwargs):
        """
        Add thermal noise to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.noise`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, **kwargs)``.
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
    def add_reflections(self, model, **kwargs):
        """
        Add auto- or cross-reflections to data visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.reflections`,
                or a callable which has the signature ``fnc(vis, fqs, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the reflections model function, other than `vis` and `fqs`. Common
                parameters are `dly`, `phs` and `amp`.
        """
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            # the following performs the modification in-place
            self.data.data_array[blt_ind, 0, :, pol_ind] = model(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                # Axis 0 is spectral windows, of which at this point there are always 1.
                freqs=self.data.freq_array[0] * 1e-9,
                **kwargs
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

    @_model()
    def add_gains(self, **kwargs):
        """
        Add gains to visibilities.

        Args:
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the gen_gains method in :mod:~`hera_sim.sigchain`.
        """

        gains = sigchain.gen_gains(
            freqs=self.data.freq_array[0] * 1e-9, ants=self.data.get_ants(), **kwargs
        )

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            self.data.data_array[blt_ind, 0, :, pol_ind] = sigchain.apply_gains(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                gains=gains,
                bl=(ant1, ant2)
            )

    @_model()
    def add_xtalk(self, **kwargs):
        """
        Add crosstalk to visibilities.

        Args:
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the gen_xtalk method in :mod:~`hera_sim.sigchain`.
        """

        xtalk = sigchain.gen_xtalk(freqs=self.data.freq_array[0] * 1e-9, **kwargs)

        # At the moment, the cross-talk function applies the same cross talk to every baseline/time.
        # Not sure if this is good or not.
        for i in range(len(self.data.get_pols())):
            self.data.data_array[:, 0, :, i] = sigchain.apply_xtalk(
                vis=self.data.data_array[:, 0, :, i], xtalk=xtalk
            )
