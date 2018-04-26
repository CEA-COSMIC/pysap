# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains linears operators classes.
"""

from modopt.opt.linear import LinearParent
from modopt.signal.wavelet import filter_convolve


class WaveletConvolve2(LinearParent):
    """Wavelet Convolution Class

    This class defines the wavelet transform operators via convolution with
    predefined filters

    Parameters
    ----------
    filters: np.ndarray
        Array of wavelet filter coefficients
    method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

    """

    def __init__(self, filters, method='astropy'):

        self._filters = filters
        self.op = lambda x: filter_convolve(x, self._filters, method=method)
        self.adj_op = lambda x: filter_convolve(x, self._filters,
                                                filter_rot=True, method=method)
