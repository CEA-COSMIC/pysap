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
import pysap.base.utils as utils

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

        Parameters
        ----------

        type_of_filtering: int
        coef_detection_method: int
        type_of_multiresolution_transform: float
        type_of_filters: float
        type_of_non_orthog_filters: float
        sigma_noise: float
        type_of_noise: int
        number_of_scales: int
        iter_max: double
        epsilon: float
        verbose: Boolean
        tab_n_sigma: ndarray
        suppress_isolated_pixels: Boolean

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

        Parameters
        ----------

        type_of_deconvolution: int
        type_of_multiresolution_transform: int
        type_of_filters: int
        number_of_undecimated_scales: int
        sigma_noise: float
        type_of_noise: int
        number_of_scales: int
        nsigma: float
        number_of_iterations: int
        epsilon: float
        psf_max_shift: bool
        verbose: bool
        optimization: bool
        fwhm_param: float
        convergence_param: float
        regul_param: float
        first_guess: string
        icf_filename: string
        rms_map: string
        kill_last_scale: bool
        positive_constraint: bool
        keep_positiv_sup: bool
        sup_isol: bool
        pas_codeur: float
        sigma_gauss: float
        mean_gauss: float
        """
        self.data = None
        self.deconv = pysparse.MRDeconvolve(**kwargs)

    def deconvolve(self, img, psf):
        """ Execute the deconvolution operation.

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


class MR2D1D():
    """ Define the structure that will be used to
        store the MR2D1D transform and reconstruction
        results.
    """
    def __init__(self, **kwargs):
        """ Define the transform.

        Parameters
        ----------

        type_of_multiresolution_transform: int
        normalize: bool
        verbose: bool
        number_of_scales_2D: int
        number_of_scales: int

        """
        self.cube = None
        self.recons = None
        self.trf = pysparse.MR2D1D(**kwargs)

    def transform(self, data):
        """ Execute the transform operation.

        Parameters
        ----------
        data: ndarray
            the input data.
        """
        self.cube = self.trf.transform(data)

    def reconstruct(self, data):
        """ Execute the reconstructiom operation.

        Parameters
        ----------
        data: ndarray
            the input data.
        """
        self.recons = self.trf.reconstruct(data)
