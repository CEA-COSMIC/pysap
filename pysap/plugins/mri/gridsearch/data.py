"""
DATA
====

Module that provide helper to load specific image.

Credit: H Cherkaoui
"""

# Sys import
import os.path as osp

import sys

# Third party import
from scipy.io import loadmat
from scipy import misc
from numpy.random import randn
import numpy as np
import scipy.fftpack as pfft

# Specific import
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.reconstruct.utils import convert_locations_to_mask
from pysap.data import get_sample_data


_dirname_ = osp.dirname(osp.abspath(__file__))
_data_dirname_ = osp.join(_dirname_, "data")
DATADIR = osp.join(osp.expanduser("~"), ".local", "share", "pysap")

_data_dirname_ = DATADIR


def _l2_normalize(x):
    """ Normalize x by its l2 norm.
    Parameters:
    -----------
    x: np.ndarray
        the input array.

    Return:
    -------
    x_norm: np.ndarray
        the l2 normalized array.
    """
    return x / np.linalg.norm(x)


def _normalize_localisations(loc):
    """ Normalize localisation to [-0.5, 0.5[.
    """
    Kmax = loc.max()
    Kmin = loc.min()
    if Kmax < np.abs(Kmin):
        return loc / (2 * np.abs(Kmin))
    else:
        loc[loc == Kmax] = -Kmax
        return loc / (2 * np.abs(Kmax))


def load_exbaboon_512_retrospection(sigma=0.0, mask_type="cartesianR4",
                                    acc_factor=None):
    """ Load the baboon's brain.

    Parameters:
    ----------
    sigma: float
        the variance of the gaussian noise added to the kspace.

    mask_type: str, (default="cartesianR4")
        the type of subsampling mask, possible choice is: 'cartesianR4',
        'radial-sparkling' or 'radial'.

    Return:
    ------
    ref: np.ndarray,
        the reference image.

    loc: np.ndarray,
        the localisation of the acquisition in the kspace.

    kspace: np.ndarray,
        the measured kspace.

    binary_mask: np.ndarray,
        the binary ROI mask of the image baboon brain.

    info: dict,
        usefull information on the characteristic of acquisition.
    """
    # ref
    # get_sample_data('mri-babouin512')
    # get_sample_data('mri-babouin512-mask')
    # get_sample_data('mri-babouin512-ref')
    # get_sample_data('mri-samples_radialx8')
    # get_sample_data('mri-samples_radialx15')
    # get_sample_data('mri-values_radialx8')
    # get_sample_data('mri-values_radialx15')
    # get_sample_data('mri-samples_sparklingx8')
    # get_sample_data('mri-samples_sparklingx15')
    # get_sample_data('mri-values_sparklingx8')
    # get_sample_data('mri-values_sparklingx15')

    imfile = "Ref_babouin_NEX32.mat"
    impath = osp.join(_data_dirname_, imfile)
    ref = _l2_normalize(loadmat(impath)['im_ref'])

    # loc, kspace
    if mask_type == "cartesianR4":
        if acc_factor is not None:
            raise ValueError("acc_factor should be None if "
                             "mask_type='cartesianR4', got "
                             "{0}".format(acc_factor))
        maskfile = "mask_BrainPhantom512_R4.mat"
        maskpath = osp.join(_data_dirname_, maskfile)
        mask = pfft.ifftshift(loadmat(maskpath)['mask'])
        loc = convert_mask_to_locations(mask)
        kspace = mask * pfft.fft2(ref)

    elif mask_type == "radial":
        if acc_factor == 8:
            maskfile = "samples_radial_x8_64x3072.mat"
            maskpath = osp.join(_data_dirname_, maskfile)
            loc = _normalize_localisations(loadmat(maskpath)['samples'])
            kspacefile = "values_radial_x8_64x3072.mat"
            kspacepath = osp.join(_data_dirname_, kspacefile)
            kspace = loadmat(kspacepath)['values']

        elif acc_factor == 15:
            maskfile = "samples_radial_x15_34x3072.mat"
            maskpath = osp.join(_data_dirname_, maskfile)
            loc = _normalize_localisations(loadmat(maskpath)['samples'])
            kspacefile = "values_radial_x15_34x3072.mat"
            kspacepath = osp.join(_data_dirname_, kspacefile)
            kspace = loadmat(kspacepath)['values']

        else:
            raise ValueError("acc_factor should be in [8, 15], got "
                             "{0}".format(acc_factor))

    elif mask_type == "radial-sparkling":
        if acc_factor == 8:
            maskfile = "samples_sparkling_x8_64x3072.mat"
            maskpath = osp.join(_data_dirname_, maskfile)
            loc = _normalize_localisations(loadmat(maskpath)['samples'])
            kspacefile = "values_sparkling_x8_64x3072.mat"
            kspacepath = osp.join(_data_dirname_, kspacefile)
            kspace = loadmat(kspacepath)['values']

        elif acc_factor == 15:
            maskfile = "samples_sparkling_x15_34x3072.mat"
            maskpath = osp.join(_data_dirname_, maskfile)
            loc = _normalize_localisations(loadmat(maskpath)['samples'])
            kspacefile = "values_sparkling_x15_34x3072.mat"
            kspacepath = osp.join(_data_dirname_, kspacefile)
            kspace = loadmat(kspacepath)['values']

        else:
            raise ValueError("acc_factor should be in [8, 15], got "
                             "{0}".format(acc_factor))

    else:
        raise ValueError("type_mask not understood, got {0} in stead of \
                         'cartesianR4', 'radial-sparkling', \
                         'radial'".format(mask_type))
    # create noise
    noise = sigma * (randn(*kspace.shape) + 1.j*randn(*kspace.shape))

    # save the noise level
    info = {'sigma': sigma}
    info['snr'] = 20.0 * np.log(np.linalg.norm(kspace) / np.linalg.norm(noise))
    info['psnr'] = 20.0*np.log(np.max(np.abs(kspace)) / np.linalg.norm(noise))

    # add noise
    kspace = kspace + noise

    # binary mask
    binarymaskfile = "Ref_N512_NEX32_mask.png"
    binarymaskpath = osp.join(_data_dirname_, binarymaskfile)
    binary_mask = ~misc.imread(binarymaskpath)[:, :, 0]
    binary_mask[binary_mask != 0] = 1

    # info
    info.update({'N': 512, 'FOV(mm)': 200, 'TE(ms)': 30, 'TR(ms)': 550,
                 'Tobs(ms)': 30.72, 'Angle(degree)': 25,
                 'Slice-thickness(mm)': 3,
                 'Contrast': 'T2*w'})
    info['mask_type'] = mask_type
    info['acc_factor'] = acc_factor

    return ref.astype("complex128"),\
        loc.astype("double"),\
        kspace.astype("complex128"),\
        np.rot90(np.fliplr(binary_mask)), info


def get_example_data(sigma=0.0, mask_type="cartesianR4",
                     acc_factor=None):

    ref = get_sample_data("mri-slice-nifti")
    ref = _l2_normalize(ref.data)
    print(ref.shape)
    mask = get_sample_data("mri-mask")
    # Generate the subsampled kspace
    kspace_mask = pfft.ifftshift(mask.data)
    kspace = pfft.fft2(ref.data) * kspace_mask

    # Get the locations of the kspace samples
    loc = convert_mask_to_locations(kspace_mask)

    # create noise
    noise = sigma * (randn(*kspace.shape) + 1.j*randn(*kspace.shape))

    # save the noise level
    info = {'sigma': sigma}
    info['snr'] = 20.0 * np.log(np.linalg.norm(kspace) / np.linalg.norm(noise))
    info['psnr'] = 20.0*np.log(np.max(np.abs(kspace)) / np.linalg.norm(noise))

    # add noise
    kspace = kspace + noise

    # binary mask
    # binarymaskfile = "Ref_N512_NEX32_mask.png"
    # binarymaskpath = osp.join(_data_dirname_, binarymaskfile)
    # binary_mask = ~misc.imread(binarymaskpath)[:, :, 0]
    # binary_mask[binary_mask != 0] = 1
    binary_mask = np.ones(ref.shape)

    # info
    info.update({'N': 512, 'FOV(mm)': 200, 'TE(ms)': 30, 'TR(ms)': 550,
                 'Tobs(ms)': 30.72, 'Angle(degree)': 25,
                 'Slice-thickness(mm)': 3,
                 'Contrast': 'T2*w'})
    info['mask_type'] = mask_type
    info['acc_factor'] = acc_factor

    return ref.astype("complex128"),\
        loc.astype("double"),\
        kspace.astype("complex128"),\
        np.rot90(np.fliplr(binary_mask)), info
