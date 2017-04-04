##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import importlib
import numpy

# Package import
from pisap.base.exceptions import Exception
from pisap.plotting import plot_wavelet


# Global parameters
# > simple wavelet factory
WMAP = {
    "haar": {
        "family_name": "Haar",
        "wtype": "orthogonal",
        "func": "pisap.sparsity.wavelet2d.haar",
        "dwt": True
    },
    "97": {
        "family_name": "Bior4",
        "wtype": "biorthogonal",
        "func": "pisap.sparsity.wavelet2d.bio97",
        "dwt": True
    }
}


class Wavelet(object):
    """ Data structure representing a wavelet transform.

    The wavelet functions must return these parameters exactly:

    * dec_hi: array: highpass decomposition.  
    * dec_lo: array: lowpass decomposition.
    * rec_hi: array: highpass reconstruction.
    * rec_lo: array: lowpass reconstruction.
    * dec_len: int: decomposition length.
    * rec_len: int: reconstruction length.
    """
    def __init__(self, name, **kwargs):
        """ Initialize the Wavelet class.

        Parameters
        ----------
        name: str
            the wavelet name.
        kwargs: dict
            the parameters that will be used when the wavlet function is
            called.
        """
        # Check input parameters
        if name not in WMAP:
            raise Exception("Unrecognize wavelet '{0}'.".format(name))

        # Define class attributes
        wavelet_info = WMAP[name]
        self.wtype = wavelet_info["wtype"]
        self.name = name
        self.family_name = wavelet_info["family_name"]
        self.dwt = wavelet_info["dwt"]
        if self.wtype not in ("orthogonal", "biorthogonal"):
            raise Exception(
                "Unrecognize wavelet type '{0}'.".format(self.wtype))

        # Import the wavelet function
        self._func = self.load(wavelet_info["func"])
        (self.dec_hi, self.dec_lo, self.rec_hi, self.rec_lo,
         self.dec_len, self.rec_len) = self._func(**kwargs)

    def __str__(self):
        """ Method that defines the string representation.
        """
        s = [
            u"Wavelet {0}".format(self.name),
            u"  Family name:    {0}".format(self.family_name),
            u"  Filters length: {0}".format(self.dec_len),
            u"  Type:           {0}".format(self.wtype),
            u"  DWT:            {0}".format(self.dwt)]
        return "\n".join(s)

    def wavefun(self, scale=5):
        """ Computes approximations of scaling function (phi) and wavelet
        function (psi) on xgrid (x) at a given scale.

        Parameters
        ----------
        scale: int (optional, default 5)
            desired scale.

        Returns
        -------
        [phi, psi, x]: array
            for orthogonal wavelets returns scaling function, wavelet function
            and xgrid - [phi, psi, x].

        [phi_d, psi_d, phi_r, psi_r, x]: array
            for biorthogonal wavelets returns scaling and wavelet function both
            for decomposition and reconstruction and xgrid.
        """
        # Avoid cycling import
        from .wtools import upcoef

        # Generate the decomposition coefficients
        coeffs = numpy.array([numpy.sqrt(2.) ** scale])

        # Orthogonal case
        if self.wtype == "orthogonal":

            # Compute extent
            output_length = (self.dec_len - 1) * 2.**scale + 1
            keep_length = self.get_keep_length(output_length, scale)
            if (output_length - keep_length - 2) < 0:
                output_length = keep_length + 2
            right_extent_length = output_length - keep_length -1

            # Compute parameters
            phi = numpy.concatenate((
                [0.],
                upcoef(coeffs, self, part="a", scale=scale, take=keep_length),
                numpy.zeros(right_extent_length)))
            psi = numpy.concatenate((
                [0.],
                upcoef(coeffs, self, part="d", scale=scale, take=keep_length),
                numpy.zeros(right_extent_length)))
            x = numpy.linspace(0., (output_length - 1) / 2.**scale,
                               output_length)

            return phi, psi, x

        # Other cases
        else:
            raise NotImplementedError("This method has only been implemented "
                                      "for 'orthogonal' wavelets.")

    def get_keep_length(self, output_length, scale):
        """ Define the data extent
        """
        lplus = self.dec_len - 2
        keep_length = 1
        for i in range(scale):
            keep_length = 2 * keep_length + lplus
        return keep_length

    def show(self, scales=None):
        """ Display approximations of scaling function (phi) and wavelet
        function (psi) on xgrid (x) at given scales.

        Parameters
        ----------
        scales: list  (optional, deafault None)
            the desired scales, if None compute at scale 4.
        """
        scales = scales or [4]
        plot_wavelet(self, scales)

    @classmethod
    def load(cls, func_repr):
        """ Load a wavelet function from its string representation.

        Parameters
        ----------
        func_repr: str
            the function to be loaded representation, ie. module1.module2.myfunc

        Returns
        -------
        func: callable
            the wavelet function.
        """
        # Split the the function representation
        func_list = func_repr.split(".")
        mod_name = ".".join(func_list[:-1])
        func_name = func_list[-1]

        # Get the function
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)

        return func


