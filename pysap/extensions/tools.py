# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Package import
from .wrapper import Sparse2dWrapper


def mr_transform(
        in_image, out_mr_file, type_of_multiresolution_transform=2,
        type_of_lifting_transform=3, number_of_scales=4,
        write_all_bands=False, write_all_bands_with_block_interp=False,
        nbiter=3, type_of_filters=1, use_l2_norm=False,
        type_of_non_orthog_filters=2, number_of_undecimated_scales=None,
        verbose=False):
    """ Wrap the Sparse2d 'mr_transform'.
    """
    # Generate the command
    cmd = [
        "mr_transform",
        "-t", type_of_multiresolution_transform,
        "-n", number_of_scales,

        "-U", type_of_non_orthog_filters]
    for key, value in [
            ("-x", write_all_bands),
            ("-B", write_all_bands_with_block_interp),
            ("-v", verbose)]:
        if value:
            cmd.append(key)
    if number_of_undecimated_scales is not None:
        cmd += ["-u", number_of_undecimated_scales]
    if type_of_multiresolution_transform in (6, 7, 8, 10, 12, 11, 27):
        cmd += ["-c", nbiter]
    if type_of_multiresolution_transform in (29, ):
        cmd += ["-l", type_of_lifting_transform]
    if type_of_multiresolution_transform in (14, ):
        cmd += ["-T", type_of_filters]
        cmd += ["-L"]
    cmd += [in_image, out_mr_file]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    process(cmd)


def mr_filter(
        in_image, out_image, type_of_filtering=1, coef_detection_method=1,
        type_of_multiresolution_transform=2, type_of_filters=1,
        type_of_non_orthog_filters=2,
        sigma=None, type_of_noise=1, number_of_scales=4,
        number_of_iterations=10, epsilon=0.001, verbose=False,
        tab_n_sigma=[], suppress_isolated_pixels=False):
    """ Wrap the Sparse2d 'mr_filter'.
    """
    # Generate the command
    cmd = ["mr_filter"]
    if type_of_noise != 1:
        cmd += ["-m", type_of_noise]
    if type_of_multiresolution_transform != 2:
        cmd += ["-t", type_of_multiresolution_transform]
    if type_of_non_orthog_filters != 2:
        cmd += ["-U", type_of_non_orthog_filters]
    if coef_detection_method != 1:
        cmd += ["-C", coef_detection_method]
    if type_of_filtering != 1:
        cmd += ["-f", type_of_filtering]
    if epsilon != 0.001:
        cmd += ["-e", epsilon]
    if number_of_iterations != 10:
        cmd += ["-i", number_of_iterations]
    if type_of_filters != 1:
        cmd += ["-T", type_of_filters]
    if tab_n_sigma != []:
        cmd += ["-s"]
        for val in tab_n_sigma:
            cmd += [val]
    if suppress_isolated_pixels:
        cmd += ["-K"]
    cmd += [in_image, out_image]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    process(cmd)


def mr_deconv(
        in_image, in_psf, out_image, type_of_deconvolution=3,
        type_of_multiresolution_transform=2, type_of_filters=1,
        number_of_undecimated_scales=None, sigma=None, c=None,
        type_of_noise=1, number_of_scales=4, nsigma=3,
        number_of_iterations=500, epsilon=0.001,
        rms_map_file_name=None, icf_fwhm=None,
        suppress_positive_constraint=False, icf_file_name=None,
        first_guess_file_name=None, residual_file_name=None,
        no_auto_shift_max_psf=False, detect_only_positive_structure=False,
        suppress_isolated_pixels=False, suppress_last_scale=False,
        optimization=None, regul_param=0, verbose=False):
    """ Wrap the Sparse2d 'mr_deconv'.
    """
    # Generate the command
    cmd = ["mr_deconv"]

    if (type_of_deconvolution != 3):
        cmd += ["-d", type_of_deconvolution]
    if (type_of_multiresolution_transform != 2):
        cmd += ["-t", type_of_multiresolution_transform]
    if (type_of_filters != 1):
        cmd += ["-T", type_of_filters]
    if (type_of_noise != 1):
        cmd += ["-m", type_of_noise]
    if (number_of_scales != 4):
        cmd += ["-n", number_of_scales]
    if (nsigma != 3):
        cmd += ["-s", nsigma]
    if (number_of_iterations != 500):
        cmd += ["-i", number_of_iterations]
    if (epsilon != 0.001):
        cmd += ["-e", epsilon]
    if (regul_param != 0):
        cmd += ["-G", regul_param]

    for key, value in [
            ("-P", suppress_positive_constraint),
            ("-S", no_auto_shift_max_psf),
            ("-p", detect_only_positive_structure),
            ("-k", suppress_isolated_pixels),
            ("-K", suppress_last_scale),
            ("-v", verbose)]:
        if value:
            cmd.append(key)
    for key, value in [
            ("-u", number_of_undecimated_scales),
            ("-g", sigma),
            ("-c", c),
            ("-R", rms_map_file_name),
            ("-f", icf_fwhm),
            ("-I", icf_file_name),
            ("-F", first_guess_file_name),
            ("-r", residual_file_name),
            ("-O", optimization)]:
        if value is not None:
            cmd += [key, value]

    cmd += [in_image, in_psf, out_image]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    process(cmd)


def mr_recons(
        in_mr_file, out_image, verbose=False):
    """ Wrap the Sparse2d 'mr_recons'.
    """
    # Generate the command
    cmd = ["mr_recons"]
    if verbose:
        cmd.append("-v")
    cmd += [in_mr_file, out_image]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    process(cmd)


def mr3d_recons(in_mr_file, out_image, verbose=False):
    """ Wrap the Sparse2d 'mr3d_recons'.
    """
    # Generate the command
    cmd = ["mr3d_recons"]
    if verbose:
        cmd.append("-v")
    cmd += [in_mr_file, out_image]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    process(cmd)


def mr3d_transform(
        in_image, out_mr_file, type_of_multiresolution_transform=2,
        type_of_lifting_transform=3, number_of_scales=4,
        type_of_filters=1, use_l2_norm=False,
        verbose=False):
    """ Wrap the Sparse2d 'mr3d_trans'.
    """
    # Generate the command
    cmd = [
        "mr3d_trans",
        "-t", type_of_multiresolution_transform,
        "-n", number_of_scales]
    for key, value in [("-v", verbose)]:
        if value:
            cmd.append(key)

    # Bi orthogonal transform
    if type_of_multiresolution_transform == 1:
        if type_of_filters == 10:
            raise ValueError("Wrong type of filters with orthogonal transform")
        if (type_of_lifting_transform != 3 and
           type_of_lifting_transform is not None):
            raise ValueError("Wrong type of lifting transform with orthogonal")
        for key, value in [("-l", type_of_lifting_transform),
                           ("-T", type_of_filters)]:
            if value is not None:
                cmd += [key, value]
        for key, value in [("-L", use_l2_norm)]:
            if value:
                cmd.append(key)

    # (bi) orthogonal transform with lifting
    if type_of_multiresolution_transform == 2:
        for key, value in [("-l", type_of_lifting_transform)]:
            if value is not None:
                cmd += [key, value]

    # A trous wavelet transform
    if type_of_multiresolution_transform == 3:
        if (type_of_lifting_transform != 3 and
           type_of_lifting_transform is not None):
            raise ValueError("Wrong type of lifting transform with orthogonal")
        for key, value in [("-l", type_of_lifting_transform)]:
            if value is not None:
                cmd += [key, value]

    cmd += [in_image, out_mr_file]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    process(cmd)


def mr3d_filter(
        in_image, out_image,
        type_of_multiresolution_transform=2, type_of_filters=1,
        sigma=None, correlated_noise=None, number_of_scales=4,
        nsigma=3,
        verbose=False):
    """ Wrap the Sparse2d 'mr3d_filter'.
    """
    # WARNING: relative path with ~  doesn't work, use absolute path from /home
    # Generate the command
    cmd = [
        "mr3d_filter",
        "-t", type_of_multiresolution_transform,
        "-T", type_of_filters,
        "-n", number_of_scales]
    for key, value in [
            ("-C", correlated_noise),
            ("-v", verbose)]:
        if value:
            cmd.append(key)
    for key, value in [
            ("-g", sigma),
            ("-s", nsigma)]:
        if value is not None:
            cmd += [key, value]
    cmd += [in_image, out_image]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    process(cmd)


def mr2d1d_trans(
        in_image, out_image,
        type_of_multiresolution_transform=14, number_of_scales_2D=5,
        number_of_scales_1D=4, normalize=False, verbose=False, reverse=False):
    """ Wrap the Sparse2d 'mr21d_trans'.
    """
    # WARNING: relative path with ~  doesn't work, use absolute path from /home
    # Generate the command
    cmd = [
        "mr2d1d_trans",
        "-t", type_of_multiresolution_transform,
        "-n", number_of_scales_2D,
        "-N", number_of_scales_1D]
    for key, value in [
            ("-M", normalize),
            ("-v", verbose),
            ("-r", reverse)]:
        if value:
            cmd.append(key)

    cmd += [in_image, out_image]

    # Execute the command
    process = Sparse2dWrapper(verbose=verbose)
    out = process(cmd)
