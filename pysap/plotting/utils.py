# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module with usefull plottinting utility functions.
"""

# Package import
import pysap

# Third party import
import numpy as np
from skimage import data, img_as_float
from skimage import exposure


def scaling(image, method="stretching"):
    """
    Change the image dynamic.

    Parameters
    ----------
    image: Image
        the image to be transformed.
    method: str, default 'stretching'
        the normalization method: 'stretching', 'equalization' or 'adaptive'.

    Returns
    -------
    normalize_image: Image
        the normalized image.
    """
    # Contrast stretching
    if method == "stretching":
        p2, p98 = np.percentile(image.data, (2, 98))
        norm_data = exposure.rescale_intensity(image.data, in_range=(p2, p98))

    # Equalization
    elif method == "equalization":
        norm_data = exposure.equalize_hist(image.data)

    # Adaptive Equalization
    elif method == "adaptive":
        norm_data = exposure.equalize_adapthist(image.data, clip_limit=0.03)

    # Unknown method
    else:
        raise ValueError("Unknown normalization '{0}'.".format(method))

    normalize_image = pysap.Image(data=norm_data)

    return normalize_image


def histogram(image, nbins=256, lower_cut=0., cumulate=0):
    """
    Compute the histogram of an input dataset.

    Parameters
    ----------
    image: Image
        the image that contains the dataset to be analysed.
    nbins: int, default 256
        the histogram number of bins.
    lower_cut: float, default 0
        do not consider the intensities under this threshold.
    cumulate: bool, default False
        if set compute the cumulate histogram.

    Returns
    -------
    hist_im: Image
        the generated histogram.
    """
    hist, bins = np.histogram(image.data[image.data > lower_cut], nbins)
    if cumulate:
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        hist_im = pysap.Image(data=cdf_normalized)
    else:
        hist_im = pysap.Image(data=hist)
    return hist_im
