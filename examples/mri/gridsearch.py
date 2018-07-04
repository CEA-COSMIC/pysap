"""
Gridsearch
==========

This example is a simplified version of the arg parser in the gridsearch module
It shows how to pass a reconstruction algorithm with a list of arguments in the
grid_search function, to multi-thread reconstructions and find best params for
each metric.

WARNING: A gridsearch can be quite long. If you have huge test dimension,
better use the combo study_launcher/post_processing tool in the plugin, which
stores results and allows you to process it afterward (read README in plugin
folder for more info)

Credit: B Sarthou
"""

# System import
import sys
import pprint as pp

# Third party import
import numpy as np
from numpy.random import randn
import scipy.fftpack as pfft
import matplotlib.pyplot as plt
from modopt.math.metrics import ssim, snr, psnr, nrmse

# Package import
from pysap.data import get_sample_data
from pysap.numerics.gridsearch import grid_search
from pysap.numerics.gridsearch import sparse_rec_condatvu
from pysap.numerics.gridsearch import sparse_rec_fista
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations


#############################################################################
# Choose global parameters and data loading
# -----------------------------------------
#
# The first parameters are linked to the data (type of sampling scheme, noise
# added to the data, etc...). The others are for the reconstructions
# Then data is loaded according to those parameters: the 2D image reference,
# the sampling locations, the k-space supposedly mesured by MRI, and a mask for
# acceleration
# The code is able to deal with cartesian, sparkling or radial kspace sampling
# trajectory. For the moment only toy cartesian data are available

# The sigma correspond to the standard deviation of the centered gaussian
# noise added to the kspace.
sigma = 0.1

# Boolean to know which reconstruction algorithm you want to try
CONDAT = True
FISTA = False

# Max number of iterations before stopping
max_nb_of_iter = 10

# Numbers of threads created on mCPU (if -1, all, if -2 all but one)
n_jobs = -2

# Verbose parameters, activated if >1
verbose_reconstruction = 0
verbose_gridsearch = 11

# Data loading
ref = get_sample_data("mri-slice-nifti")
ref.data = ref.data / np.linalg.norm(ref.data)
mask = get_sample_data("mri-mask")

# Generate the subsampled kspace
kspace_mask = pfft.ifftshift(mask.data)
kspace = pfft.fft2(ref.data) * kspace_mask

# Get the locations of the kspace samples
loc = convert_mask_to_locations(kspace_mask)

# Create noise
noise = sigma * (randn(*kspace.shape) + 1.j * randn(*kspace.shape))

# Add noise
kspace = kspace + noise

# Binary mask
binmask = np.ones(ref.shape)


#############################################################################
# Declaration of metrics
# ----------------------
#
# We declare in a dictionnary which metric to compute during gridsearch, along
# differents parameters associated to it, especially if we want to check early
# stopping on it

metrics = {
    "ssim": {
        "metric": ssim,
        "mapping": {"x_new": "test", "y_new": None},
        "cst_kwargs": {"ref": ref, "mask": binmask},
        "early_stopping": True},
    "psnr": {
        "metric": psnr,
        "mapping": {"x_new": "test", "y_new": None},
        "cst_kwargs": {"ref": ref, "mask": binmask},
        "early_stopping": False},
    "snr": {
        "metric": snr,
        "mapping": {"x_new": "test", "y_new": None},
        "cst_kwargs": {"ref": ref, "mask": binmask},
        "early_stopping": False},
    "nrmse": {
        "metric": nrmse,
        "mapping": {"x_new": "test", "y_new": None},
        "cst_kwargs": {"ref": ref, "mask": binmask},
        "early_stopping": False}
}


#############################################################################
# Declaration of parameters to gridsearch
# ---------------------------------------
#
# We declare lists of parameters to be gridsearch (every one except the data
# and the wavelets can be gridsearched)

mu_list = list(np.logspace(-8, -1, 5))
nb_scales = [3, 4]
list_wts = [
    "MallatWaveletTransform79Filters",
    "UndecimatedBiOrthogonalTransform"]


#############################################################################
# Gridsearch
# ----------
#
# For each wavelet and each reconstruction algorithm, we call the gridsearch
# which will execute each reconstruction on each set of parameters
# (multi-threaded, to take advantage of mCPU configuration) and then return the
# resulted best fit according to each metric

# Go through all transform
ssim_metric = {}
for wt in list_wts:

    print("Using wavelet {0}".format(wt))

    # Case Condat
    if CONDAT:

        # Params
        params = {
            "data": kspace,
            "wavelet_name": wt,
            "samples": loc,
            "nb_scales": nb_scales,
            "mu": mu_list,
            "max_nb_of_iter": max_nb_of_iter,
            "sigma": 0.1,
            "metrics": metrics,
            "verbose": verbose_reconstruction,
        }

        # Launch the gridsearch
        list_kwargs, results = grid_search(sparse_rec_condatvu,
                                           params, n_jobs=n_jobs,
                                           verbose=verbose_gridsearch)

    # Case FISTA
    elif FISTA:

        # Params
        params = {
            "data": kspace,
            "wavelet_name": wt,
            "samples": loc,
            "nb_scales": nb_scales,
            "mu": mu_list,
            "max_nb_of_iter": max_nb_of_iter,
            "metrics": metrics,
            "verbose": verbose_reconstruction,
        }

        # Launcher the gridsearch
        list_kwargs, results = grid_search(sparse_rec_fista,
                                           params, n_jobs=n_jobs,
                                           verbose=verbose_gridsearch)

    else:
        print("No reconstruction called.")

    # Reorganize the gridsearch outputs

    for res, params in zip(results, list_kwargs):
        key = "mu={0}-scales={1}-transform={2}".format(
            params["mu"], params["nb_scales"], params["wavelet_name"])
        ssim_metric[key] = res[2]["ssim"]["values"][-1]


#############################################################################
# Results
# -------
#
# Simply dipslay the SSIM metrics to choose the best set of parameters.

pp.pprint(ssim_metric)
fig, ax = plt.subplots()
ssims = ssim_metric.values()
ax.set_title("SSIM metric over the optimized parameters.")
ax.plot(range(len(ssims)), ssims, "o", ms=20, lw=2, alpha=0.7, mfc="orange")
ax.grid()
plt.show()
