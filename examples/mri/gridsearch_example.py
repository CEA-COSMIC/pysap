"""
EX grid_search
====

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

# Sys import
import sys

import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

from pysap.plugins.mri.gridsearch.data import load_exbaboon_512_retrospection
from pysap.plugins.mri.gridsearch.study_launcher import _gather_result
from pysap.base.gridsearch import grid_search
from pysap.plugins.mri.gridsearch.reconstruct_gridsearch import *

from modopt.math.metrics import ssim, snr, psnr, nrmse

#############################################################################
# Choose global parameters and data loading
# -------------------
#
# The first parameters are linked to the data (type of sampling scheme, noise
# added to the data, etc...). The others are for the reconstructions
# Then data is loaded according to those parameters: the 2D image reference,
# the sampling locations, the k-space supposedly mesured by MRI, and a mask for
# acceleration


# Be careful: the name of the kspace sampling trajectory should be one of the
# available in the data module: cartesianR4, sparkling or radial.
mask_type = 'cartesianR4'

# the available acceleration factor dependent of the choice of the mask type,
# if there is no need for that option write None. (options: None, 8, 15)
acc_factor = None

# the sigma correspond to the standard deviation of the centered gaussian
# noise added to the kspace.
sigma = 0.1

# Boolean to know which reconstruction algorithm you want to try
CONDAT = True
FISTA = False
# Max number of iterations before stopping
max_nb_of_iter = 200
# Numbers of threads created on mCPU (if -1, all, if -2 all but one)
n_jobs = 16
# Verbose parameters, activated if >1
verbose_reconstruction = 11
verbose_gridsearch = 11

# data loading
res = load_exbaboon_512_retrospection(sigma, mask_type, acc_factor)
ref, loc, kspace, binmask, info = res[0], res[1], res[2], res[3], res[4]

#############################################################################
# Declaration of metrics
# -------------------
#
# We declare in a dictionnary which metric to compute during gridsearch, along
# differents parameters associated to it, especially if we want to check early
# stopping on it


metrics = {'ssim': {'metric': ssim,
                    'mapping': {'x_new': 'test', 'y_new': None},
                    'cst_kwargs': {'ref': ref, 'mask': binmask},
                    'early_stopping': True,
                    },
           'psnr': {'metric': psnr,
                    'mapping': {'x_new': 'test', 'y_new': None},
                    'cst_kwargs': {'ref': ref, 'mask': binmask},
                    'early_stopping': False,
                    },
           'snr': {'metric': snr,
                   'mapping': {'x_new': 'test', 'y_new': None},
                   'cst_kwargs': {'ref': ref, 'mask': binmask},
                   'early_stopping': False,
                   },
           'nrmse': {'metric': nrmse,
                     'mapping': {'x_new': 'test', 'y_new': None},
                     'cst_kwargs': {'ref': ref, 'mask': binmask},
                     'early_stopping': False,
                     },
           }

#############################################################################
# Declaration of parameters to gridsearch
# -------------------
#
# We declare lists of parameters to be gridsearch (every one except the data
# and the wavelets can be gridsearched)

mu_list = list(np.logspace(-8, -1, 5))
nb_scales = [3, 4]
list_wts = ["MallatWaveletTransform79Filters",
            "UndecimatedBiOrthogonalTransform",
            ]

#############################################################################
# Gridsearch
# -------------------
#
# For each wavelet and each reconstruction algorithm, we call the gridsearch
# which will execute each reconstruction on each set of parameters
# (multi-threaded, to take advantage of mCPU configuration) and then return the
# resulted best fit according to each metric


for wt in list_wts:

    print("Using wavelet {0}".format(wt))

    if CONDAT:
        # Params Condat
        params = {
            'data': kspace,
            'wavelet_name': wt,
            'samples': loc,
            'nb_scales': nb_scales,
            'mu': mu_list,
            'max_nb_of_iter': max_nb_of_iter,
            'sigma': 0.1,
            'metrics': metrics,
            'verbose': verbose_reconstruction,
        }

        # launch the gridsearch
        list_kwargs, results = grid_search(sparse_rec_condatvu,
                                           params, n_jobs=n_jobs,
                                           verbose=verbose_gridsearch)

        # gather the best result per metric
        best_res_condat = {'ssim': _gather_result(metric='ssim',
                                                  metric_direction=True,
                                                  list_kwargs=list_kwargs,
                                                  results=results),
                           'snr': _gather_result(metric='snr',
                                                 metric_direction=True,
                                                 list_kwargs=list_kwargs,
                                                 results=results),
                           'psnr': _gather_result(metric='psnr',
                                                  metric_direction=True,
                                                  list_kwargs=list_kwargs,
                                                  results=results),
                           'nrmse': _gather_result(metric='nrmse',
                                                   metric_direction=False,
                                                   list_kwargs=list_kwargs,
                                                   results=results),
                           }

    elif FISTA:
        # Params FISTA
        params = {
            'data': kspace,
            'wavelet_name': wt,
            'samples': loc,
            'nb_scales': nb_scales,
            'mu': mu_list,
            'max_nb_of_iter': max_nb_of_iter,
            'metrics': metrics,
            'verbose': verbose_reconstruction,
        }
        # launcher the gridsearch
        list_kwargs, results = grid_search(sparse_rec_fista,
                                           params, n_jobs=n_jobs,
                                           verbose=verbose_gridsearch)

        # gather the best result per metric
        best_res_fista = {'ssim': _gather_result(metric='ssim',
                                                 metric_direction=True,
                                                 list_kwargs=list_kwargs,
                                                 results=results),
                          'snr': _gather_result(metric='snr',
                                                metric_direction=True,
                                                list_kwargs=list_kwargs,
                                                results=results),
                          'psnr': _gather_result(metric='psnr',
                                                 metric_direction=True,
                                                 list_kwargs=list_kwargs,
                                                 results=results),
                          'nrmse': _gather_result(metric='nrmse',
                                                  metric_direction=False,
                                                  list_kwargs=list_kwargs,
                                                  results=results),
                          }
    else:
        print('No reconstruction called')

#############################################################################
# Results
# -------------------
#
# We have a dictionnary for each reconstruction algorithm. In each, is stored a
# a key for each metric,in which there is a dictionnary storing the best value
# for the metric, the best set of parameters according to the metric and the
# reconstruction data associated with those parameters. See the pretty print
# below to see the dict structure.


pp.pprint(best_res_condat)
print('Best set of parameters for Condat algorithm, for SSIM metric:\n',
      'best params:', best_res_condat['ssim']['best_params'], '\n',
      'best_transform:', best_res_condat['ssim']['best_result'][1])

coef = best_res_condat['ssim']['best_result'][0]
img = np.abs(coef)
fig = plt.figure(figsize=(18, 13))
fig.suptitle('Best result for SSIM')

ax = fig.add_subplot(1, 2, 1)
ax.matshow(np.abs(img)[140:350, 100:325], cmap='gray')
ax.set_title("ssim = {0}".format(ssim(img, ref, binmask)))

ax = fig.add_subplot(1, 2, 2)
ax.matshow(np.abs(ref)[140:350, 100:325], cmap='gray')
ax.set_title('Reference')

plt.show()
