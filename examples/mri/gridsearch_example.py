"""
EX grid_search
====

Module that provide helper to load specific image.
Scripts are extracted from study_launcher.py and post_processing.py
in the gridsearch plugin.
It parses the config.ini file to get lists of parameters to test on a specific
reconstruction algorithm and store results (numerics and statistics with the
post-processing part) in results/ dir.
The scripts are explained better in the plugin README. 

Credit: B Sarthou
"""



# Sys import
import sys
import os
import logging
import argparse

sys.path.insert(0,'/home/bs255482/src/Modopt/ModOpt/')

from pysap import info
from pysap.plugins.mri.reconstruct.linear import Wavelet2 as Wavelet
from pysap.plugins.mri.gridsearch.data import load_exbaboon_512_retrospection
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations, convert_locations_to_mask

from pysap.plugins.mri.gridsearch.study_launcher import _launch
from pysap.plugins.mri.gridsearch.post_processing import _main, _save_ref
from pysap.plugins.mri.gridsearch.post_processing import _coherence, _get_and_save_best_image
from pysap.plugins.mri.gridsearch.post_processing import _save_sparsity_images, _wavelets_runtimes

from pysap.utils import TempDir

if sys.version_info[0] < 3:
    import ConfigParser
else:
    import configparser as ConfigParser

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

DEFAULT_EMAIL = 'bsarthou@hotmail.fr'

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


if __name__ == '__main__':
    with TempDir(isap='results') as tmpd:
        parser = argparse.ArgumentParser(description=''.join(__doc__))
        parser.add_argument('-o', '--output-dir', dest='root_dirname',
                            action ='store_const',
                            const  ='results',
                            default='results',
                            help   ='root directoy of the results')
        parser.add_argument("-v", "--verbose", help="increase output verbosity",
                            action="store_true")
        parser.add_argument("--do-email-report", help="send a report email",
                            dest='emailreport', action="store_true")
        parser.add_argument('--email-dest', dest='emaildest',
                            action='store_const',
                            const=DEFAULT_EMAIL,
                            default=DEFAULT_EMAIL,
                            help='set the email destination')
        args = parser.parse_args()

        if args.verbose:
            logging.info(info())

        if not os.path.exists(args.root_dirname):
            os.makedirs(args.root_dirname)

        config = ConfigParser.RawConfigParser()
        config.read('config.ini')

        # gathe the global params for the study
        global_params = dict(config.items('Global'))
        global_params['n_jobs'] = int(global_params['n_jobs'])
        global_params['timeout'] = int(global_params['timeout'])
        global_params['verbose_reconstruction'] = bool(global_params['verbose_reconstruction'])
        global_params['verbose_gridsearch'] = bool(global_params['verbose_gridsearch'])
        global_params['max_nb_of_iter'] = int(global_params['max_nb_of_iter'])

        global_params['verbose_reconstruction'] = True
        global_params['verbose_gridsearch'] = True

        # gather the run specific params and launch the run
        for section in config.sections():
            if "Run" in section:
                params = dict(config.items(section))
                params.update(global_params)
                try:
                    params['acc_factor'] = float(params['acc_factor'])
                except ValueError:
                    params['acc_factor'] = None
                sigma_list = params['sigma'].split('[')[1].split(']')[0].split(',')
                sigma_list = [float(value) for value in sigma_list]
                params['dirname'] = os.path.join(args.root_dirname,
                                                 params['mask_type'])
                if not os.path.exists(params['dirname']):
                    os.makedirs(params['dirname'])

                for sigma in sigma_list:
                    params['sigma'] = sigma
                    _launch(**params)

        # Post- processing of results
        ROOT_DIR = 'results/cartesianR4'
        OUT_DIR = 'results/cartesianR4'
        VERBOSE = True

        # metric plots generation
        _main(ROOT_DIR, OUT_DIR, VERBOSE)

        # save ref
        _save_ref(OUT_DIR)

        # coherence computation
        _, loc, _, _, _ = load_exbaboon_512_retrospection()
        level = 5 # don't change
        wts_list = ["UndecimatedBiOrthogonalTransform",
                    "MallatWaveletTransform79Filters",
                    ]
        coherence_list = []
        for wt_name in wts_list:
            coherence_list.append(_coherence(Wavelet(wt_name, level), loc))
        coherence_disp = ["{0}: {1}\n".format(wt, round(coh, 4))
                              for coh, wt in sorted(zip(coherence_list, wts_list))]
        coherence_disp.insert(0, "Coherences:\n")
        coherence_file = os.path.join(OUT_DIR, "coherences.txt")
        with open(coherence_file, 'w') as pfile:
            pfile.writelines(coherence_disp)

        # resulting images selection and display
        mask_type = "radial-sparkling"
        wts_type = "*"
        acc_factor = "8"
        sigma = "0.0"
        output_path = os.path.join(OUT_DIR, "image_illustration", "all_wts")
        _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)

        mask_type = "radial-sparkling"
        wts_type = "UndecimatedBiOrthogonalTransform"
        acc_factor = "*"
        sigma = "0.0"
        output_path = os.path.join(OUT_DIR, "image_illustration", "UndeBiOrtho")
        _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)
        mask_type = "radial"
        wts_type = "UndecimatedBiOrthogonalTransform"
        acc_factor = "*"
        sigma = "0.0"
        output_path = os.path.join(OUT_DIR, "image_illustration", "UndeBiOrtho")
        _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)
        mask_type = "cartesianR4"
        wts_type = "UndecimatedBiOrthogonalTransform"
        acc_factor = "None"
        sigma = "0.0"
        output_path = os.path.join(OUT_DIR, "image_illustration", "UndeBiOrtho")
        _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)
        mask_type = "cartesianR4"
        wts_type = "UndecimatedBiOrthogonalTransform"
        acc_factor = "None"
        sigma = "0.8"
        output_path = os.path.join(OUT_DIR, "image_illustration", "UndeBiOrtho")
        _get_and_save_best_image(mask_type, wts_type, acc_factor, sigma, output_path)

        # sparsity computation
        thresholding_method_list = ["manual_l2_based_threshold",
                                    "manual_threshold",
                                    "histogram_threshold",
                                    ]

        for thresholding_method in thresholding_method_list:
            _save_sparsity_images(thresholding_method, OUT_DIR)

        # runtime time computation
        nb_op = 10
        wt_names = ["UndecimatedBiOrthogonalTransform",
                    "FastCurveletTransform"]
        timings = _wavelets_runtimes(wt_names, nb_op=nb_op)
        timings_repr = ["{0}: {1} s\n".format(wt, t) for t, wt in sorted(zip(timings, wt_names))]
        timings_repr.insert(0, "CPU runtimes average on {0} decomposition and recomposition:\n".format(nb_op))
        timing_file = os.path.join(OUT_DIR, "runtimes.txt")
        with open(timing_file, 'w') as pfile:
            pfile.writelines(timings_repr)
