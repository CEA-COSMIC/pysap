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
        type_of_non_orthog_filters=2, number_of_undecimated_scales=None,
        sigma=None, c=None, type_of_noise=1, number_of_scales=4,
        nsigma=3, number_of_iterations=10, epsilon=0.001,
        support_file_name=False, suppress_isolated_pixels=False,
        suppress_last_scale=False, detect_only_positive_structure=False,
        E=1e-3, size_block=7, niter_sigma_clip=1, first_detection_scale=1,
        rms_map_file_name=None, suppress_positive_constraint=False,
        add_maximum_level_constraint=None, background_model_image=False,
        flat_image=False, use_second_generation_filter=False,
        consider_missing_data=False, mask_file_name=None,
        write_info_probability_map=False, regul_param=0.1, snr_file_name=None,
        verbose=False):
    """ Wrap the Sparse2d 'mr_filter'.
    """
    # Generate the command
    cmd = [
        "mr_filter",
        "-f", type_of_filtering,
        "-C", coef_detection_method,
        "-t", type_of_multiresolution_transform,
        "-T", type_of_filters,
        "-U", type_of_non_orthog_filters,
        "-m", type_of_noise,
        "-n", number_of_scales,
        "-s", nsigma,
        "-i", number_of_iterations,
        "-e", epsilon,
        "-E", E,
        "-S", size_block,
        "-N", niter_sigma_clip,
        "-F", first_detection_scale,
        "-G", regul_param]
    for key, value in [
            ("-w", support_file_name),
            ("-k", suppress_isolated_pixels),
            ("-K", suppress_last_scale),
            ("-p", detect_only_positive_structure),
            ("-P", suppress_positive_constraint),
            ("-B", background_model_image),
            ("-M", flat_image),
            ("-A", use_second_generation_filter),
            ("-H", consider_missing_data),
            ("-h", write_info_probability_map),
            ("-v", verbose)]:
        if value:
            cmd.append(key)
    for key, value in [
            ("-u", number_of_undecimated_scales),
            ("-g", sigma),
            ("-c", c),
            ("-R", rms_map_file_name),
            ("-b", add_maximum_level_constraint),
            ("-I", mask_file_name),
            ("-Q", snr_file_name)]:
        if value is not None:
            cmd += [key, value]
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
    cmd = [
        "mr_deconv",
        "-d", type_of_deconvolution,
        "-t", type_of_multiresolution_transform,
        "-T", type_of_filters,
        "-m", type_of_noise,
        "-n", number_of_scales,
        "-s", nsigma,
        "-i", number_of_iterations,
        "-e", epsilon,
        "-G", regul_param]
    for key, value in [
            ("-P", suppress_positive_constraint),
            ("-S", do_not_auto_shift_max_psf),
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
