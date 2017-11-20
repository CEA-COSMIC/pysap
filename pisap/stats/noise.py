# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np

# Package import
import pisap


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


def multiscale_sigma_mad(grad_op, linear_op):
    """ Estimate the std from the mad routine on each approximation scale.

    Parameters
    ----------
    grad_op: instance
        Gradient operator.
    linear_op: instance
        Linear operator.
    Returns
    -------
    sigma: list
        a list of str estimate for each scale.
    """
    d = linear_op.op(grad_op.grad)
    sigma = []
    for scale in range(linear_op.transform.nb_scale):
        scale_data = linear_op.transform[scale]
        if isinstance(scale_data, list):
            scale_data = np.concatenate(scale_data)
        sigma.append(sigma_mad(scale_data))
    return sigma


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
        hist_im = pisap.Image(data=cdf_normalized)
    else:
        hist_im = pisap.Image(data=hist)
    return hist_im
