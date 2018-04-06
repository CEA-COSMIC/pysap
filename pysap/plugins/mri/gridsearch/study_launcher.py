#!/usr/bin/python

"""
LAUNCHER
========

Script to launcher the study from the EUSIPCO paper material.

Credit: H Cherkaoui
"""

# Sys import
import sys
import os
import logging

# sys.path.remove('/home/bs255482/.local/lib/python3.5/site-packages/modopt-1.1.4-py3.5.egg')
sys.path.insert(0,'/home/bs255482/src/Modopt/ModOpt/')

import argparse
import itertools

# Third party import
import requests
import numpy as np
import matplotlib.pylab as plt
import pickle
import pprint

from pysap.base.gridsearch import grid_search

from pysap import info
from pysap.plugins.mri.gridsearch.reconstruct_gridsearch import sparse_rec_condatvu
from pysap.plugins.mri.gridsearch.reconstruct_gridsearch import sparse_rec_fista

from pysap.plugins.mri.reconstruct.gradient import GradAnalysis2 as Grad2DAnalysis
from pysap.plugins.mri.reconstruct.linear import Wavelet2 as Wavelet
from pysap.plugins.mri.reconstruct.fourier import FFT2 as FFT
from pysap.plugins.mri.reconstruct.fourier import NFFT2 as NFFT

from modopt.opt.metrics import ssim, snr, psnr,nrmse

# local import
from pysap.plugins.mri.gridsearch.data import load_exbaboon_512_retrospection

if sys.version_info[0] < 3:
	import ConfigParser
else:
	import configparser as ConfigParser

DEFAULT_EMAIL = 'bsarthou@hotmail.fr'

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def _gather_result(metric, metric_direction, list_kwargs, results):
	""" Gather the best reconstruction result.

	Parameters:
	-----------
	metric: str,
		the name of the metric, it will become a dict key in the ouput dict.
	metric_direction: bool,
		if True the higher the better the metric value is, else the lower the
		better.
	list_kwargs: list of dict,
		the list of kwargs of the gridsearch
	results: list of list,
		list of the result of the gridsearch

	Return:
	-------
	gathered_results: dict,
		the gatheres result: the best value of the metric, the best set of
		parameters, the best reconstructed data.
	"""
	list_metric = []
	for res in results:
		list_metric.append(res[2][metric]['values'][-1])

	list_metric = np.array(list_metric)

	# get best runs
	if metric_direction:
		best_metric = list_metric.max()
		best_idx = list_metric.argmax()
	else:
		best_metric = list_metric.min()
		best_idx = list_metric.argmin()
	tmp_params = list_kwargs[best_idx]
	best_params = {}

	# reduce params kwargs
	best_params['max_nb_of_iter'] = tmp_params['max_nb_of_iter']
	best_params['mu'] = tmp_params['mu']
	try:
		best_params['sigma'] = tmp_params['sigma']
	except KeyError:  # in case of fista run
		pass
	best_result = results[best_idx]

	return {'best_value': best_metric, 'best_params': best_params,
			'best_result': best_result}


def _launch(sigma, mask_type, acc_factor, dirname, max_nb_of_iter, n_jobs,
			timeout, verbose_reconstruction, verbose_gridsearch):
	""" Launch a grid search to the specific given data.
	"""
	# data loading
	res = load_exbaboon_512_retrospection(sigma, mask_type, acc_factor)
	ref, loc, kspace, binmask, info = res[0], res[1], res[2], res[3], res[4]
	logging.info("Data information:\n\n{0}\n".format(pprint.pformat(info)))

	# metric declaration
	metrics = {'ssim': {'metric':ssim,
					   'mapping': {'x_new': 'test', 'y_new':None},
					   'cst_kwargs':{'ref':ref, 'mask':binmask},
					   'early_stopping': True, # early-stop on ssim
					   },
			   'psnr': {'metric':psnr,
					   'mapping': {'x_new': 'test', 'y_new':None},
					   'cst_kwargs': {'ref':ref, 'mask':binmask},
					   'early_stopping': False,
					   },
			   'snr': {'metric':snr,
					  'mapping': {'x_new': 'test', 'y_new':None},
					  'cst_kwargs': {'ref':ref, 'mask':binmask},
					  'early_stopping': False,
					  },
			   'nrmse':{'metric':nrmse,
						'mapping': {'x_new': 'test', 'y_new':None},
						'cst_kwargs':{'ref':ref, 'mask':binmask},
						'early_stopping': False,
						},
			   }

	# # principal gridsearch params grid
	# mu_list = list(np.logspace(-8, -1, 20))
	# nb_scales = [3, 4, 5]
	# list_wts = ["MallatWaveletTransform79Filters",
	# 			"UndecimatedBiOrthogonalTransform",
	# 			"MeyerWaveletsCompactInFourierSpace",
	# 			"BsplineWaveletTransformATrousAlgorithm",
	# 			"FastCurveletTransform",
	# 			]

	# params tests
	mu_list = list(np.logspace(-8, -1, 5))
	nb_scales = [3, 4]
	list_wts = ["MallatWaveletTransform79Filters",
				"UndecimatedBiOrthogonalTransform",
				]


	for wt in list_wts:

		logging.info("Using wavelet {0}".format(wt))

		wt_list = [{'nb_scale': nb_reso, 'wavelet': wt} for nb_reso in nb_scales]
		ft_cls = NFFT if mask_type in ['radial-sparkling', 'radial'] else FFT
		ft_cls_kwargs = {ft_cls: {"samples_locations": loc,
								  "img_size": ref.shape[0]}
						}
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

		# launcher the gridsearch
		list_kwargs, results = grid_search(sparse_rec_condatvu,
		params, n_jobs=n_jobs,
		verbose=verbose_gridsearch)

		# #Params FISTA
		# params = {
		# 	'data': kspace,
		# 	'wavelet_name': wt,
		# 	'samples': loc,
		# 	'nb_scales': nb_scales,
		# 	'mu': mu_list,
		# 	'max_nb_of_iter': max_nb_of_iter,
		# 	'metrics': metrics,
		# 	'verbose': verbose_reconstruction,
		# }
		# # launcher the gridsearch
		# list_kwargs, results = grid_search(sparse_rec_fista,
		# 								   params, n_jobs=n_jobs,
		# 								   verbose=verbose_gridsearch)

		# gather the best result per metric
		best_results = {'ssim': _gather_result(metric='ssim',
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

		# save the gathered results
		wt_dirname = os.path.join(dirname, wt)
		if not os.path.exists(wt_dirname):
			os.makedirs(wt_dirname)
		filename = ("study__{0}_{1}_{2}__{3}.pkl").format(mask_type, sigma,
														  acc_factor, wt)
		filepath = os.path.join(wt_dirname, filename)
		with open(filepath, 'wb') as pfile:
			pickle.dump(best_results, pfile)


# if __name__ == '__main__':
#
# 	parser = argparse.ArgumentParser(description=''.join(__doc__))
# 	parser.add_argument('-o', '--output-dir', dest='root_dirname',
# 						action ='store_const',
# 						const  ='results',
# 						default='results',
# 						help   ='root directoy of the results')
# 	parser.add_argument("-v", "--verbose", help="increase output verbosity",
# 						action="store_true")
# 	parser.add_argument("--do-email-report", help="send a report email",
# 						dest='emailreport', action="store_true")
# 	parser.add_argument('--email-dest', dest='emaildest',
# 						action='store_const',
# 						const=DEFAULT_EMAIL,
# 						default=DEFAULT_EMAIL,
# 						help='set the email destination')
# 	args = parser.parse_args()
#
# 	if args.verbose:
# 		logging.info(info())
#
# 	if not os.path.exists(args.root_dirname):
# 		os.makedirs(args.root_dirname)
#
# 	config = ConfigParser.RawConfigParser()
# 	config.read('config.ini')
#
# 	# gathe the global params for the study
# 	global_params = dict(config.items('Global'))
# 	global_params['n_jobs'] = int(global_params['n_jobs'])
# 	global_params['timeout'] = int(global_params['timeout'])
# 	global_params['verbose_reconstruction'] = bool(global_params['verbose_reconstruction'])
# 	global_params['verbose_gridsearch'] = bool(global_params['verbose_gridsearch'])
# 	global_params['max_nb_of_iter'] = int(global_params['max_nb_of_iter'])
#
# 	global_params['verbose_reconstruction'] = True
# 	global_params['verbose_gridsearch'] = True
#
# 	# gather the run specific params and launch the run
# 	for section in config.sections():
# 		if "Run" in section:
# 			params = dict(config.items(section))
# 			params.update(global_params)
# 			try:
# 				params['acc_factor'] = float(params['acc_factor'])
# 			except ValueError:
# 				params['acc_factor'] = None
# 			sigma_list = params['sigma'].split('[')[1].split(']')[0].split(',')
# 			sigma_list = [float(value) for value in sigma_list]
# 			params['dirname'] = os.path.join(args.root_dirname,
# 											 params['mask_type'])
# 			if not os.path.exists(params['dirname']):
# 				os.makedirs(params['dirname'])
#
# 			for sigma in sigma_list:
# 				params['sigma'] = sigma
# 				_launch(**params)
