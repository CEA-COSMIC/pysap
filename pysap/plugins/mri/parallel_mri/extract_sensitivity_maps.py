# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains tools to extract sensitivity maps from undersampled MR
acquisition with high density in the k space center.
"""
# System import

# Package import
from scipy.interpolate import griddata
import scipy.fftpack as pfft

# Third party import
import numpy as np


def extract_k_space_center(samples, samples_locations,
                           thr=None, img_shape=None):
    """
    This class extract the k space center for a given threshold or it estimate
    the threshold if any is provided using the density of the sampling scheme
    and the resolution.

    Parameters
    ----------
    samples: np.ndarray
        The value of the samples
    samples_locations: np.ndarray
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
    thr: float
        The threshold used to extract the k_space center
    img_shape: tuple
        The image shape to estimate the cartesian density

    Returns
    -------
    The extracted center of the k-space
    """
    if thr is None:
        if img_shape is None:
            raise ValueError('target image cartesian image shape must be fill')
        raise NotImplementedError
    else:
        samples_thresholded = np.copy(samples)
        samples_thresholded *= (samples_locations[:, 0] <= thr)
        samples_thresholded *= (samples_locations[:, 1] <= thr)
    return samples_thresholded


def gridding_2d(points, values, img_shape, method='linear'):
    """
    Interpolate non-Cartesian data into a cartesian grid

    Parameters
    ----------
    points: np.ndarray
        The 2D k_space locations of size [M, 2]
    values: np.ndarray
        An image of size [N_x, N_y]
    img_shape: tuple
        The final output ndarray
    method: {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation for more details see scipy.interpolate.griddata
        documentation

    Returns
    -------
    np.ndarray
        The gridded solution of shape [N_x, N_y]
    """
    xi = np.linspace(np.min(points),
                     np.max(points),
                     img_shape[0],
                     endpoint=False)
    yi = np.linspace(np.min(points),
                     np.max(points),
                     img_shape[1],
                     endpoint=False)
    grid_x, grid_y = np.meshgrid(xi, yi)
    return griddata(points,
                    values,
                    (grid_x, grid_y),
                    method=method,
                    fill_value=0)


def get_Smaps(k_space, img_shape, samples=None, mode='Gridding'):
    """
    This method estimate the sensitivity maps information from parallel mri
    acquisition and for variable density sampling scheme where teh k-space
    center had been heavily sampled.

    Parameters
    ----------
    k_space: np.ndarray
        The acquired kspace of shape (M,L), where M is the number of samples
        acquired and L is the number of coils used
    samples: np.ndarray

    Returns
    -------
    Smaps: np.ndarray
        the estimated sensitivity maps of shape (img_shape, L) with L the
        number of channels
    ISOS: np.ndarray
        The sum of Squarre used to extract the sensitivity maps
    """
    if samples is None:
        mode = 'FFT'

    M, L = k_space.shape
    Smaps_shape = (img_shape[0], img_shape[1], L)
    Smaps = np.zeros(Smaps_shape).astype('complex128')
    if mode == 'FFT':
        if not M == img_shape[0]*img_shape[1]:
            raise ValueError(['The number of samples in the k-space must be',
                              'equal to the (image size, the number of coils)'
                              ])
        k_space = k_space.reshape(Smaps_shape)
        for l in range(Smaps_shape[2]):
            Smaps[:, :, l] = pfft.ifftshift(pfft.ifft2(k_space[:, :, l]))
    elif mode == 'NFFT':
        raise ValueError('NotImplemented yet')
    else:
        xi = np.linspace(0, img_shape[0], endpoint=False)
        yi = np.linspace(0, img_shape[1], endpoint=False)
        gridx, gridy = np.meshgrid(xi, yi)
        for l in range(L):
            Smaps[:, :, l] = pfft.ifftshift(
                                            pfft.ifft2(griddata(
                                                samples,
                                                k_space[:, l],
                                                (gridx, gridy),
                                                method='linear',
                                                fill_value=0)))

    SOS = np.sqrt(np.sum(np.abs(Smaps)**2, axis=2))
    for r in range(L):
        Smaps[:, :, r] /= SOS
    return Smaps, SOS
