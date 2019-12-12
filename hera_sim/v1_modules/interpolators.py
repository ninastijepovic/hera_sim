"""
This module provides interfaces to different interpolation classes.
"""

import numpy as np
from cached_property import cached_property
from scipy.interpolate import RectBivariateSpline, interp1d
from hera_sim.data import DATA_PATH
from os import path

INTERP_OBJECTS = {"1d": ("beam", "bandpass",),
                  "2d": ("Tsky_mdl", ) }

def _check_path(datafile):
    # if the datafile is not an absolute path, assume it's in the data folder
    if not path.isabs(datafile):
        datafile = path.join(DATA_PATH, datafile)
    # make sure that the path exists
    assert path.exists(datafile), \
            "If datafile is not an absolute path, then it is assumed to " \
            "exist in the hera_sim.data folder. The datafile passed could " \
            "not be found; please ensure that the path to the file exists"
    return datafile

def _read_npy(npy):
    return np.load(_check_path(npy))

class Interpolator:
    """Base interpolator class"""

    def __init__(self, datafile, **interp_kwargs):
        """Initialize an `Interpolator` object with necessary attributes.

        Parameters
        ----------
        datafile : str
            Path to the file to be used to generate the interpolation object.
            Must be either a .npy or .npz file, depending on which type of
            interpolation object is desired. If path is not absolute, then the 
            file is assumed to exist in the `data` directory of `hera_sim` and 
            is modified to reflect this assumption.

        interp_kwargs : unpacked dict, optional
            Passed to the interpolation method used to make the interpolator.
        """
        self._datafile = _check_path(datafile)
        self._data = np.load(self._datafile, allow_pickle=True)
        self._interp_kwargs = interp_kwargs

class Tsky(Interpolator):
    """Sky temperature interpolator; subclass of `Interpolator`."""

    def __init__(self, datafile, **interp_kwargs):
        """Extend the `Interpolator` constructor.

        Parameters
        ----------
        datafile : str
            Passed to superclass constructor. Must be a `.npz` file with the
            following archives:
                'tsky':
                    Array of sky temperature values in units of Kelvin; must
                    have shape=(NPOLS, NLSTS, NFREQS).

                'freqs':
                    Array of frequencies at which the tsky model is evaluated,
                    in units of GHz; must have shape=(NFREQS,).

                'lsts':
                    Array of LSTs at which the tsky model is evaulated, in
                    units of radians; must have shape=(NLSTS,).

                'meta':
                    Dictionary of metadata describing the data stored in the npz
                    file. Currently it only needs to contain an entry 'pols', 
                    which lists the polarizations such that their order agrees
                    with the ordering of arrays along the tsky axis-0. The user
                    may choose to also save the units of the frequency, lst, and
                    tsky arrays as strings in this dictionary.

        interp_kwargs : unpacked dict, optional
            Extend interp_kwargs parameter for superclass to allow for the 
            specification of which polarization to use via the key 'pol'. If 
            'pol' is specified, then it must be one of the polarizations listed 
            in the 'meta' dictionary.

        Attributes
        ----------
        freqs : np.ndarray
            Frequency array used to construct the interpolator object. Has 
            units of GHz and shape=(NFREQS,).

        lsts : np.ndarray
            LST array used to construct the interpolator object. Has units of 
            radians and shape=(NLSTS,).

        tsky : np.ndarray
            Sky temperature array used to construct the interpolator object. 
            Has units of Kelvin and shape=(NPOLS, NLSTS, NFREQS).

        meta : dict
            Dictionary containing some metadata relevant to the interpolator.

        pol : str, default 'xx'
            Polarization appropriate for the sky temperature model. Must be 
            one of the polarizations stored in the 'meta' dictionary.

        Raises
        ------

        AssertionError: 
            Raised if any of the required npz keys are not found or if the 
            tsky array does not have shape=(NPOLS, NLSTS, NFREQS).
        """
        super().__init__(self, datafile, **interp_kwargs)
        self._check_npz_format()
        self.pol = self._interp_kwargs.pop("pol", "xx")
        self._check_pol(self.pol)

    def __call__(self, lsts, freqs):
        """Evaluate the Tsky model at the specified lsts and freqs."""
        return self._interpolator(lsts, freqs)

    @property
    def freqs(self):
        return self._data['freqs']

    @property
    def tsky(self):
        return self._data['tsky']

    @property
    def lsts(self):
        return self._data['lsts']

    @property
    def meta(self):
        return self._data['meta'][None][0]

    @cached_property
    def _interpolator(self):
        """Construct an interpolation object.
        
        Uses class attributes to construct an interpolator using the 
        scipy.interpolate.RectBivariateSpline interpolation class.
        """
        # get index of tsky's 0-axis corresponding to pol
        pol_index = self.meta['pols'].index(self.pol)
        
        # get the tsky data
        tsky_data = self.tsky[pol_index]

        # do some wrapping in LST
        lsts = np.concatenate([self.lsts[-10:]-2*np.pi,
                               self.lsts,
                               self.lsts[:10]+2*np.pi])
        tsky_data = np.concatenate([tsky_data[-10:], tsky_data, tsky_data[:10]])

        # now make the interpolation object
        return RectBivariateSpline(lsts, self.freqs, tsky_data, **self._interp_kwargs)

    def _check_npz_format(self):
        """Check that the npz archive is formatted properly."""

        assert 'freqs' in self._data.keys(), \
                "The frequencies corresponding to the sky temperature array " \
                "must be provided. They must be saved to the npz file using " \
                "the key 'freqs'."
        assert 'lsts' in self._data.keys(), \
                "The LSTs corresponding to the sky temperature array must " \
                "be provided. They must be saved to the npz file using the " \
                "key 'lsts'."
        assert 'tsky' in self._data.keys(), \
                "The sky temperature array must be saved to the npz file " \
                "using the key 'tsky'."
        assert 'meta' in self._data.keys(), \
                "The npz file must contain a metadata dictionary that can " \
                "be accessed with the key 'meta'. This dictionary should " \
                "provide information about the units of the various arrays " \
                "and the polarizations of the sky temperature array."
        
        # check that tsky has the correct shape
        assert self.tsky.shape==(len(self.meta['pols']),
                                 self.lsts.size, self.freqs.size), \
                "The tsky array is incorrectly shaped. Please ensure that " \
                "the tsky array has shape (NPOLS, NLSTS, NFREQS)."

    def _check_pol(self, pol):
        """Check that the desired polarization is in the meta dict."""
        assert pol in self.meta['pols'], \
                "Polarization must be in the metadata's polarization tuple. " \
                "The metadata contains the following polarizations: " \
                "{}".format(self.meta['pols'])

class FreqInterpolator(Interpolator):
    """Frequency interpolator; subclass of `Interpolator`."""
    def __init__(self, datafile, **interp_kwargs):
        """Extend the `Interpolator` constructor.

        Parameters
        ----------
        datafile : str
            Passed to the superclass constructor.

        interp_kwargs : unpacked dict, optional
            Extends superclass interp_kwargs parameter by checking for the key
            'interpolator' in the dictionary. The 'interpolator' key should 
            have the value 'poly1d' or 'interp1d'; these correspond to the 
            `np.poly1d` and `scipy.interpolate.interp1d` objects, respectively.
            If the 'interpolator' key is not found, then it is assumed that 
            a `np.poly1d` object is to be used for the interpolator object.

        Raises
        ------
        AssertionError:
            This is raised if the choice of interpolator and the required type
            of the ref_file do not agree (i.e. trying to make a 'poly1d' object
            using a .npz file as a reference). An AssertionError is also raised
            if the .npz for generating an 'interp1d' object does not have the
            correct arrays in its archive.
        """
        super().__init__(self, datafile, **interp_kwargs)
        self._interp_type = self._interp_kwargs.pop("interpolator", "poly1d")
        self._obj = None

    def __call__(self, freqs):
        """Evaluate the interpolation object at the given frequencies."""
        return self._interpolator(freqs)

    @cached_property
    def _interpolator(self):
        """Construct the interpolator object."""
        if self._interp_type=='poly1d':
            return np.poly1d(self._data)
        else:
            # if not using poly1d, then need to get some parameters for
            # making the interp1d object
            obj = self._data[self._obj]
            freqs = self._data['freqs']

            # use a cubic spline by default, but override this if the user
            # specifies a different kind of interpolator
            kind = self._interp_kwargs.pop('kind', 'cubic')
            return interp1d(freqs, obj, kind=kind, **self._interp_kwargs)

    def _check_format(self):
        """Check that class attributes are appropriately formatted."""
        assert self._interp_type in ('poly1d', 'interp1d'), \
                "Interpolator choice must either be 'poly1d' or 'interp1d'."

        if self._interp_type=='interp1d':
            assert path.splitext(self._datafile)[1] == '.npz', \
                    "In order to use an 'interp1d' object, the reference file " \
                    "must be a '.npz' file."
            assert self._obj in self._data.keys() and 'freqs' in self._data.keys(), \
                    "You've chosen to use an interp1d object for modeling the " \
                    "{}. Please ensure that the `.npz` archive has the following " \
                    "keys: 'freqs', '{}'".format(self._obj, self._obj)
        else:
            # we can relax this a bit and allow for users to also pass a npz
            # with the same keys as in the above case, but it seems silly to
            # use a polynomial interpolator instead of a spline interpolator in
            # this case
            assert path.splitext(self._datafile)[1] == '.npy', \
                    "In order to use a 'poly1d' object, the reference file " \
                    "must be a .npy file that contains the coefficients for " \
                    "the polynomial fit in decreasing order."

class Beam(FreqInterpolator):
    """Beam interpolation object; subclass of `FreqInterpolator`."""
    def __init__(self, datafile, **interp_kwargs):
        """Extend the `FreqInterpolator` constructor.

        Parameters
        ----------
        datafile : str
            Passed to the superclass constructor.

        interp_kwargs : unpacked dict, optional
            Passed to the superclass constructor.
        """
        super().__init__(self, datafile, **interp_kwargs)
        self._obj = "beam"
        self._check_format()

class Bandpass(FreqInterpolator):
    """Bandpass interpolation object; subclass of `FreqInterpolator`."""
    def __init__(self, datafile, **interp_kwargs):
        """Extend the `FreqInterpolator` constructor.
        
        Parameters
        ----------
        datafile : str
            Passed to the superclass constructor.

        interp_kwargs : unpacked dict, optional
            Passed to the superclass constructor.
        """
        super().__init__(self, datafile, **interp_kwargs)
        self._obj = "bandpass"
        self._check_format()

