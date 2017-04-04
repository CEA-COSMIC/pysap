# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
#
#:Author: Samuel Farrens <samuel.farrens@gmail.com>
#:Version: 1.1
#:Date: 05/01/2017
##########################################################################

# System import
import numpy as np


def mad(data):
    """ Function that returns the median absolute deviation (MAD) of an input
    array.

    The median absolute deviation is a measure of statistical dispersion.
    Moreover, the MAD is a robust statistic, being more resilient to outliers 
    in a data set than the standard deviation. In the standard deviation,
    the distances from the mean are squared, so large deviations are weighted
    more heavily, and thus outliers can heavily influence it. In the MAD,
    the deviations of a small number of outliers are irrelevant.

    In order to use the MAD as a consistent estimator for the estimation of
    the standard deviation std, one takes std = k MAD,
    where k is a constant scale factor, which depends on the distribution.
    For normally distributed data k is taken to be 1.4826.

    Parameters
    ----------
    data: ndarray
        the input data.

    Returns
    -------
    mad: float
        the median absolute deviation.
    """
    return np.median(np.abs(data - np.median(data)))


def sigma_mad(data):
    """ Function that returns an estimation of the std using the MAD as a
    consistent estimator for the estimation of the standard deviation:
    std = k MAD,
    where k is a constant scale factor, which depends on the distribution.
    For normally distributed data k is taken to be 1.4826.
    """
    return 1.4826 * mad(data)
