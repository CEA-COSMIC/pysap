#!/usr/bin/python
"""
POST-PROCESSING:
===============

Command line tool to produce the EUSIPCO paper material.

Credit: H Cherkaoui
"""

# Sys import
import os
import sys
import time
import argparse
import logging
import tempfile
import pickle
from glob import glob

import numpy as np
import scipy.fftpack as pfft
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

# Third party import
from pisap.numerics.linear import Wavelet
from pisap.numerics.cost import ssim
from pisap.utils import convert_locations_to_mask, convert_mask_to_locations

# Local import
from data import load_exbaboon_512_retrospection

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def _get_metrics(dirname, verbose=False):
    """ Recursively collect the metric points under dirname and produce bar
    plots or plots.
    """
    mask_type_subdirs = glob(os.path.join(dirname, "*"))
    M = {}
    for mask_type_subdir in mask_type_subdirs:
        if verbose:
            mask_name = os.path.basename(mask_type_subdir)
            logging.info("Found '{0}' mask type.".format(mask_name))
        wts_subdirs = glob(os.path.join(mask_type_subdir, "*"))
        for wts_subdir in wts_subdirs:
            wt = os.path.basename(wts_subdir)
            noise_levels_files = glob(os.path.join(wts_subdir, "*.pkl"))
            for noise_levels_file in noise_levels_files:
                # parse sigma, acc_factor
                tmp = noise_levels_file.split('__')[1]
                _, sigma, acc_factor = tmp.split('_')
                sigma = float(sigma)
                try:
                    acc_factor = float(acc_factor)
                except ValueError: # case acc_factor = 'None'
                    acc_factor =  None

                mask = os.path.basename(mask_type_subdir)

                # check if the key is defined
                if mask not in M:
                    M[mask] = {}
                if acc_factor not in M[mask]:
                    M[mask][acc_factor] = {}
                with open(noise_levels_file, 'r') as pfile:
                    report = pickle.load(pfile)

                for metric_name, metric in report.items():
                    if metric_name not in M[mask][acc_factor]:
                        M[mask][acc_factor][metric_name] = {}
                    if wt not in M[mask][acc_factor][metric_name]:
                        M[mask][acc_factor][metric_name][wt] = {}
                    if sigma not in M[mask][acc_factor][metric_name][wt]:
                        M[mask][acc_factor][metric_name][wt][sigma] = {}

                    # remove all the saved metrics
                    metric['best_result'] = [metric['best_result'][0],
                                             metric['best_result'][1]]

                    M[mask][acc_factor][metric_name][wt][sigma] = metric

    return M


def _plot_metrics(M, output_dir, verbose=False):
    """ Save in .png the plots or bar plots of the metrics.
    """
    line_type = ["*-", "^-", "o-", "s-", "x-"]
    for mask_name, mask in M.items():
        for acc_factor_name, acc_factor in mask.items():
            for metric_name, metric in acc_factor.items():
                fig, ax = plt.subplots(figsize=(15, 15))
                for idx, (wt_name, wts) in enumerate(metric.items()):
                    x_values = np.array(list(wts))
                    y_values = np.array([report['best_value']
                                        for report in wts.values()])

                    do_barplot = True if (len(x_values) == 1) else False

                    order = np.argsort(x_values)
                    y_values = y_values[order]
                    # x_values are only use when plotting (and not barplot)
                    x_values = x_values[order]
                    new_x_values = []
                    for x in x_values:
                        data = load_exbaboon_512_retrospection(sigma= x,
                                                               mask_type=mask_name,
                                                               acc_factor=acc_factor_name)
                        _, _, _, _, info = data
                        new_x_values.append(info['snr'])
                    x_values = np.array(new_x_values)

                    if verbose and do_barplot:
                        logging.warning("In '{0}, {1}, {2}' loop: single "
                                        "metric value per wavelet detected: "
                                        "fallback to"
                                        "barplot.".format(mask_name,
                                                          acc_factor_name,
                                                          metric_name))

                    if do_barplot:
                        xx = range(len(y_values))
                        ax.bar(xx, y_values)
                        ax.set_xticks(xx)
                        ax.set_xticklabels(metric.iterkeys(), rotation=45)
                    else:
                        ax.plot(x_values, y_values, line_type[idx],
                                linewidth=2.0, markersize=10.0, label=wt_name)

                plt.xlabel("SNR of the input kspace", fontsize=20)
                plt.ylabel(metric_name, fontsize=20)
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, ncol=3, fontsize=16)
                acc_factor_type = acc_factor_name
                if acc_factor_type is None:
                    acc_factor_type = 4 # cartesian case
                plt.title(("Mask type: {0}, "
                           "acc. factor = {1}").format(mask_name,
                                                       acc_factor_type),
                           fontsize=35)

                plot_type = "bars" if do_barplot else "plots"
                filename = "{0}_{1}_{2}_{3}.png".format(mask_name,
                                                        acc_factor_type,
                                                        metric_name, plot_type)
                output_dir = '.' if output_dir is None else output_dir
                plots_dir = os.path.join(output_dir, "metric_plots")
                if not os.path.isdir(plots_dir):
                        os.makedirs(plots_dir)
                if verbose:
                    logging.info("plots will be saved at '{0}'".format(plots_dir))

                filepath = os.path.join(plots_dir, filename)
                fig.savefig(filepath)
                plt.close(fig)


def _main(dirname, output_dir, verbose=False):
    """ Main function of Mikado.
    """
    if verbose:
        logging.info("Mikado run on directory '{}'".format(dirname))
    metrics = _get_metrics(dirname, verbose)
    _plot_metrics(metrics, output_dir, verbose)


def _f_atom_generator_from_loc(loc, p_size):
    """ Yield the Fourier's atom present in the sampling scheme loc.
    """
    mask = convert_locations_to_mask(loc, p_size)
    x, y = np.where(mask != 0)
    for coord in zip(x, y):
        f_atom = np.zeros((p_size, p_size), dtype=np.int)
        f_atom[coord] = 1
        yield pfft.ifft2(f_atom)


def _coherence(wt, loc, p_size=10):
    """ Compute the coherence between the wavelet and the sampling scheme.
    """
    f_atoms = _f_atom_generator_from_loc(loc, p_size)
    nb_f_atoms = loc.shape[0]
    max_coh = 0.0
    for f_atom in f_atoms:
        coh = np.max(np.abs(wt.op(f_atom)))
        if max_coh < coh:
            max_coh = coh
    return max_coh


def _get_filename(mask_type, wts_type, acc_factor, sigma="*", root_dir="output_results"):
    """
    """
    try:
        acc_factor = str(float(acc_factor))
    except ValueError:
        pass
    acc_factor = "*" + sigma + "*_" + acc_factor + "_*"
    regex = os.path.join(root_dir, mask_type, wts_type, acc_factor)
    return glob(regex)


def _save_best_image(imgs_filename_list, output_dir):
    """
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in imgs_filename_list:
        with open(filename, "r") as pfile:
            report = pickle.load(pfile)
            img = np.abs(report['ssim']['best_result'][0].data)[140:350,100:325]
            ssim = round(report['ssim']['best_value'], 2)
            plt.matshow(img, cmap='gray')
            base_filename = os.path.basename(filename)
            wt_name = base_filename.split('_')[-1].split('.')[0]
            plt.title("{0}\nssim: {1}".format(wt_name, ssim))
            plt.savefig(os.path.join(output_dir, "{0}_SSIM_{1}.png".format(base_filename, ssim)))


def _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma="*", output_path="."):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename_list = _get_filename(mask_type, wts_type, acc_factor, sigma)
    _save_best_image(filename_list, output_path)
    with open(os.path.join(output_path, "readme.txt"), "w") as pfile:
        pfile.writelines(["Caracteristic\n"
                          "\tmask_type: {0}\n".format(mask_type),
                          "\twts_type: {0}\n".format(wts_type),
                          "\tacc_factor: {0}\n".format(acc_factor),
                          "\tsigma: {0}\n".format(sigma),
                          ])


def _wavelets_runtimes(wt_list, nb_scale=3, nb_op=10):
    """
    """
    ref, _, _, _, _ = load_exbaboon_512_retrospection()
    timings = []
    for wt_name in wt_list:
        wt = Wavelet(nb_scale=nb_scale, wavelet=wt_name)
        time_s = 0.0
        for i in range(nb_op):
            t0 = time.clock()
            coef = wt.op(ref)
            img = wt.adj_op(coef)
            time_s += (time.clock() - t0) / nb_op
        timings.append(time_s)
    return timings


def _save_sparsity_images(thresholding_method, output_dirname):
    """
    """
    nb_scales = range(2, 6)
    ref, _, _, binary_mask, _ = load_exbaboon_512_retrospection()
    wts_list = ["UndecimatedBiOrthogonalTransform",
                "FastCurveletTransform",
                "BsplineWaveletTransformATrousAlgorithm",
                "MallatWaveletTransform79Filters",
                "MeyerWaveletsCompactInFourierSpace",
                ]

    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    output_dir = thresholding_method + "_plots"
    plots_dir = os.path.join(output_dirname, "sparsity", output_dir)
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    for nb_scale in nb_scales:
        fig = plt.figure(figsize=(18, 13))
        fig.suptitle("Sparsity on {0} level of resolution".format(nb_scale),
                     fontsize=20)

        for idx, wt_name in enumerate(wts_list):

            # compute the wavelet coefficients
            wt = Wavelet(nb_scale=nb_scale, wavelet=wt_name)
            coef = wt.op(ref)

            fake_data = np.zeros(wt.transform._data_shape, dtype=np.double)
            bands, _ = wt.transform.trf.transform(fake_data, save=False)
            bands_shapes = [band.shape for band in bands]
            approx_length = bands_shapes[-1][0] * bands_shapes[-1][1]

            coef_no_approx = coef[-approx_length:]

            if thresholding_method == "histogram_threshold":
                card, values = np.histogram(np.abs(coef_no_approx), 100)
                centroids, mask, _ = k_means(values.flatten()[:, None], 2)
                t = max(values[mask])

            if thresholding_method == "manual_l2_based_threshold":
                l2 = np.linalg.norm(coef_no_approx)
                t = l2 * 2.5e-4

            if thresholding_method == "manual_threshold":
                t = 1.0e-4

            # compute the sparsity
            nb_coef = len(coef)
            coef[np.abs(coef) <= t] = 0
            nb_no_zeros = len(coef[np.abs(coef) > t])
            ratio = round(nb_no_zeros / float(nb_coef), 5)

            # disp
            img = wt.adj_op(coef)
            ax = fig.add_subplot(2, 3, idx+1)
            ax.matshow(np.abs(img)[140:350,100:325], cmap='gray')
            ax.set_title("{0}\nr = {1}/{2} = {3}%\nssim = {4}".format(wt_name, nb_no_zeros,
                                                          nb_coef, 100.*ratio, ssim(img, ref, binary_mask)))

        ax = fig.add_subplot(2, 3, 6)
        ax.matshow(np.abs(ref)[140:350,100:325], cmap='gray')
        ax.set_title("Reference")
        plt.savefig(os.path.join(plots_dir, "sparsity_{0}_.png".format(nb_scale)))


def _save_ref(output_dir):
    """
    """
    ref, _, _, _, _ = load_exbaboon_512_retrospection()
    img = np.abs(ref)[140:350,100:325]
    plt.matshow(img, cmap='gray')
    plt.title("Reference")
    plt.savefig(os.path.join(output_dir, "reference.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=''.join(__doc__))
    parser.add_argument('root_dirname', type=str, metavar='INPUT',
                        help='root directoy of the results')
    parser.add_argument('output_dirname', type=str, nargs='?',
                        metavar='OUTPUT', help='root directory for the plots')
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.info(__doc__)

    # metric plots generation
    _main(args.root_dirname, args.output_dirname, args.verbose)

    # save ref
    _save_ref(args.output_dirname)

    # coherence computation
    _, loc, _, _, _ = load_exbaboon_512_retrospection()
    level = 5 # don't change
    wts_list = ["UndecimatedBiOrthogonalTransform",
                "FastCurveletTransform",
                "BsplineWaveletTransformATrousAlgorithm",
                "MallatWaveletTransform79Filters",
                "MeyerWaveletsCompactInFourierSpace",
                ]
    coherence_list = []
    for wt_name in wts_list:
        coherence_list.append(_coherence(Wavelet(wt_name, level), loc))
    coherence_disp = ["{0}: {1}\n".format(wt, round(coh, 4))
                          for coh, wt in sorted(zip(coherence_list, wts_list))]
    coherence_disp.insert(0, "Coherences:\n")
    coherence_file = os.path.join(args.output_dirname, "coherences.txt")
    with open(coherence_file, 'w') as pfile:
        pfile.writelines(coherence_disp)

    # resulting images selection and display
    mask_type = "radial-sparkling"
    wts_type = "*"
    acc_factor = "8"
    sigma = "0.0"
    output_path = os.path.join(args.output_dirname, "image_illustration", "all_wts")
    _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)

    mask_type = "radial-sparkling"
    wts_type = "UndecimatedBiOrthogonalTransform"
    acc_factor = "*"
    sigma = "0.0"
    output_path = os.path.join(args.output_dirname, "image_illustration", "UndeBiOrtho")
    _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)
    mask_type = "radial"
    wts_type = "UndecimatedBiOrthogonalTransform"
    acc_factor = "*"
    sigma = "0.0"
    output_path = os.path.join(args.output_dirname, "image_illustration", "UndeBiOrtho")
    _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)
    mask_type = "cartesianR4"
    wts_type = "UndecimatedBiOrthogonalTransform"
    acc_factor = "None"
    sigma = "0.0"
    output_path = os.path.join(args.output_dirname, "image_illustration", "UndeBiOrtho")
    _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)
    mask_type = "cartesianR4"
    wts_type = "UndecimatedBiOrthogonalTransform"
    acc_factor = "None"
    sigma = "0.8"
    output_path = os.path.join(args.output_dirname, "image_illustration", "UndeBiOrtho")
    _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)

    # sparsity computation
    thresholding_method_list = ["manual_l2_based_threshold",
                                "manual_threshold",
                                "histogram_threshold",
                                ]

    for thresholding_method in thresholding_method_list:
        _save_sparsity_images(thresholding_method, args.output_dirname)

    # runtime time computation
    nb_op = 10
    wt_names = ["UndecimatedBiOrthogonalTransform",
                "FastCurveletTransform"]
    timings = _wavelets_runtimes(wt_names, nb_op=nb_op)
    timings_repr = ["{0}: {1} s\n".format(wt, t) for t, wt in sorted(zip(timings, wt_names))]
    timings_repr.insert(0, "CPU runtimes average on {0} decomposition and recomposition:\n".format(nb_op))
    timing_file = os.path.join(args.output_dirname, "runtimes.txt")
    with open(timing_file, 'w') as pfile:
        pfile.writelines(timings_repr)
