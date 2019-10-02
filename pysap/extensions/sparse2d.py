# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import

import warnings

# Package import
import pysap

from pysap.base.transform import MetaRegister  # for the metaclass
from pysap.base import image

try:
    import pysparse
except ImportError:  # pragma: no cover
    warnings.warn("Sparse2d python bindings not found, use binaries.")
    pysparse = None

# Third party import
import numpy as np
import matplotlib.pyplot as plt


class Filter():
    """ Define the structure that will be used to store the filter result.
    """
    def __init__(self, **kwargs):
        """ Define the filter.
        """
        self.data = None
        self.flt = pysparse.MRFilters(**kwargs)

    def filter(self, data):
        """ Execute the filter operation.

        Parameters
        ----------
        data: ndarray
            the input data.
        """
        self.data = pysap.Image(data=self.flt.filter(data))

    def show(self):  # pragma: no cover
        """ Show the filtered data.
        """
        if self.data is None:
            raise AttributeError("The data must be filtered first !")
        self.data.show()


class Deconvolve():
    """ Define the structure that will be used to
        store the deconvolution result.
    """
    def __init__(self, **kwargs):
        """ Define the deconvolution.
        """
        self.data = None
        self.deconv = pysparse.MRDeconvolve(**kwargs)

    def deconvolve(self, img, psf):
        """ Execute the filter operation.

        Parameters
        ----------
        img: ndarray
            the input image.
        psf: ndarray
            the input psf
        """
        self.data = pysap.Image(data=self.deconv.deconvolve(img, psf))

    def show(self):  # pragma: no cover
        """ Show the deconvolved data.
        """
        if self.data is None:
            raise AttributeError("The data must be deconvolved first !")
        self.data.show()
