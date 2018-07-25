##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Grid search method that helps launching multiple reconstructions at once.
"""

# System import
import sys
import itertools
import psutil

# Package import
from pysap.numerics.utils import generate_operators

# Third party import
from joblib import Parallel, delayed
import numpy as np


def _default_wrapper(recons_func, **kwargs):
    """ Default wrapper to parallelize the image reconstruction.
    """
    gradient_space = kwargs.pop("gradient_space")
    gradient_op, linear_op, prox_op, cost_op = generate_operators(
        data=kwargs.pop("data"),
        wavelet_name=kwargs.pop("wavelet_name"),
        samples=kwargs.pop("samples"),
        nb_scales=kwargs.pop("nb_scales"),
        non_cartesian=kwargs.pop("non_cartesian"),
        uniform_data_shape=kwargs.pop("uniform_data_shape"),
        gradient_space=gradient_space)
    if gradient_space == "analysis":
        prox_name = "prox_dual_op"
    else:
        prox_name = "prox_op"
    for name, oper in (
            ("gradient_op", gradient_op),
            ("linear_op", linear_op),
            (prox_name, prox_op),
            ("cost_op", cost_op)):
        kwargs[name] = oper
    return recons_func(**kwargs),


def grid_search(func, param_grid, wrapper=None, n_jobs=1, verbose=0):
    """ Run `func` on the carthesian product of `param_grid`.

    Parameters:
    -----------
    func: function
        The reconstruction function from whom to tune the hyperparameters.
        `func` return should be handle by wrapper if it's not a
        simple np.ndarray image.
    param_grid: dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values: the grids spanned by each
        dictionary in the list are explored.
    wrapper: function, (default: None)
        Handle the call of func if some pre-process or post-process
        should be done. `wrapper` has a specific API:
        `wrapper(func, **kwargs)`
    n_jobs: int (default: 1)
        The maximum number of concurrently running jobs, such as the number
        of Python worker processes when backend=multiprocessing or the
        size of the thread-pool when backend=threading. If -1 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2,
        all CPUs but one are used.
    verbose: int (default: 0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout. The frequency of the
        messages increases with the verbosity level. If it more than 10,
        all iterations are reported.

    Results:
    --------
    metrics: dict
        the gridsearch results. Each key corresponds to a gridsearch parameters
        set. The values are 'params' the reconstruction parameters, 'image'
        the reconstructed image, and 'metrics' the different metrics as
        specified in the input parameters.
    """
    # Define the default wrapper call function if necessary
    if wrapper is None:
        wrapper = _default_wrapper

    # Sanitize wrapper value to list type
    for key, value in param_grid.items():
        if not isinstance(value, list):
            param_grid[key] = [value]
    list_kwargs = [dict(zip(param_grid, x))
                   for x in itertools.product(*param_grid.values())]

    # Run the reconstruction with joblib
    if verbose > 0:
        if n_jobs == -1:
            n_jobs_used = psutil.cpu_count()
        elif n_jobs == -2:
            n_jobs_used = psutil.cpu_count() - 1
        else:
            n_jobs_used = n_jobs
        print(("Running grid_search for {0} candidates"
               " on {1} jobs").format(len(list_kwargs), n_jobs_used))
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                       delayed(wrapper)(func, **kwargs)
                       for kwargs in list_kwargs)

    # Reorganize the gridsearch outputs
    metrics = {}
    for cnt, (res, params) in enumerate(zip(results, list_kwargs)):
        params.pop("metrics")
        params.pop("samples")
        metrics["grid{0}".format(cnt + 1)] = {
            "params": params,
            "image": res[0][0],
            "metrics": res[0][3]}

    return metrics
