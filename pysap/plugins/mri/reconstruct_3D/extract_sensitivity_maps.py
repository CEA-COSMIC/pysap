# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
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
from joblib import Parallel, delayed
import scipy.fftpack as pfft
from copy import deepcopy
# Third party import
import numpy as np


def extract_k_space_center(data_value, samples_locations,
                           thr=None, img_shape=None):
    """
    This class extract the k-space center for a given threshold or it estimate
    the threshold if any is provided using the density of the sampling scheme
    and the resolution.
    Parameters:
    ----------
    samples: np.ndarray
        The value of the samples
    samples_locations: np.ndarray
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
    thr: float
        The threshold used to extract the k_space center
    img_shape: tuple
        The image shape to estimate the cartesian density
    Returns:
    -------
    The extracted center of the k-space
    """
    if thr is None:
        if img_shape is None:
            raise ValueError('Target image cartesian image shape must be fill')
        raise NotImplementedError
    else:
        center_locations = np.copy(samples_locations)
        data_thresholded = np.copy(data_value)
        condition = np.logical_and(
                        np.logical_and(np.abs(samples_locations[:,0])<= thr[0],
                                       np.abs(samples_locations[:,1])<= thr[1]),
                        np.abs(samples_locations[:,2])<= thr[2]
                        )
        index = np.linspace(0, samples_locations.shape[0] - 1,
                             samples_locations.shape[0]).astype('int')
        index = np.extract(condition, index)
        center_locations = samples_locations[index, :]
        data_thresholded = data_thresholded[: , index]
        return center_locations, data_thresholded

def gridding_3d (points, values, xi, method='linear', fill_value=0):
    return griddata(points=np.copy(points), values=np.copy(values), xi=deepcopy(xi),
                            method=method, fill_value=fill_value)


def get_3D_smaps(k_space, img_shape, samples=None, mode='gridding',
                 samples_min=None, samples_max=None, n_cpu=1):
    """
    This method estimate the sensitivity maps information from parallel mri
    acquisition and for variable density sampling scheme where the k-space
    center had been heavily sampled in a 3D setting.
    Parameters:
    ----------
    k_space: np.ndarray
        The acquired kspace of shape (M,L), where M is the number of samples
        acquired and L is the number of coils used
    img_shape: a 3 element tuple
        target image shape
    mode: string
        The extraction mode either: 'gridding', 'FFT', or 'NFFT'
    Returns:
    -------
    Smaps: np.ndarray
        the estimated sensitivity maps of shape (img_shape, L) with L the
        number of channels
    ISOS: np.ndarray
        The sum of Squarre used to extract the sensitivity maps
    """
    if samples_min is None:
        samples_min = [np.min(samples[:,idx]) for idx in range(samples.shape[1])]
    if samples_max is None:
        samples_max = [np.max(samples[:,idx]) for idx in range(samples.shape[1])]

    if samples is None:
        mode = 'FFT'

    L, M = k_space.shape
    Smaps = []
    if mode == 'FFT':
        if not M == img_shape[0]*img_shape[1]*img_shape[2]:
            raise ValueError(['The number of samples in the k-space must be',
                              'equal to the (image size, the number of coils)'
                              ])
        k_space = k_space.reshape(L, *img_shape)
        for l in range(L):
            Smaps.append(pfft.fftshift(pfft.ifftn(pfft.ifftshift(k_space[l]))))
    elif mode == 'NFFT':
        raise ValueError('NotImplemented yet')
    else:
        xi = np.linspace(samples_min[0],
                         samples_max[0],
                         num=img_shape[0],
                         endpoint=False)
        yi = np.linspace(samples_min[1],
                         samples_max[1],
                         num=img_shape[1],
                         endpoint=False)
        zi = np.linspace(samples_min[2],
                         samples_max[2],
                         num=img_shape[2],
                         endpoint=False)
        gridx, gridy, gridz = np.meshgrid(xi, yi, zi)

        gridded_kspaces = Parallel(n_jobs=n_cpu, verbose=1000)(delayed(gridding_3d)
            (points=np.copy(samples),
            values=np.copy(k_space[l]),
            xi=(gridx, gridy, gridz),
            method='linear',
            fill_value=0) for l in range(L))

        for gridded_kspace in gridded_kspaces:
            Smaps.append(np.swapaxes(pfft.fftshift(
                pfft.ifftn(pfft.ifftshift(gridded_kspace))), 1, 0))

    Smaps = np.asarray(Smaps)
    SOS = np.squeeze(np.sqrt(np.sum(np.abs(Smaps)**2, axis=0)))
    for l in range(L):
        Smaps[l] /= SOS
    return Smaps, SOS
