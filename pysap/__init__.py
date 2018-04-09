##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
pySap is a Python package related to sparsity and its application in
astronomical data analysis. This package is based on the 'sparse2d' C++ library
that allows sparse decomposition, denoising and deconvolution.
"""

# import matplotlib
# matplotlib.use("Agg")

from __future__ import print_function
from .info import __version__
import pysap.extensions
from pysap.base import io
from pysap.utils import TempDir
from pysap.configure import info
from pysap.base.image import Image
from pysap.utils import load_transform
from pysap.base.utils import monkeypatch
from pysap.utils import AVAILABLE_TRANSFORMS

# import sys
# # sys.path.remove('/home/bs255482/.local/lib/python3.5/site-packages/modopt-1.1.4-py3.5.egg')
# sys.path.insert(0,'/home/bs255482/src/Modopt/ModOpt/')

# Apply some monkeypatchs to the optimization package
import progressbar
from modopt.opt.algorithms import Condat
from modopt.opt.algorithms import ForwardBackward


# Display a welcome message
print(info())

# Monkey patch with alternative version of Modopt

@monkeypatch(ForwardBackward)
def iterate(self, max_iter=150):
    r"""Iterate

    This method calls update until either convergence criteria is met or
    the maximum number of iterations is reached

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of iterations (default is ``150``)

    """
    with progressbar.ProgressBar(redirect_stdout=True,
                                 max_value=max_iter) as bar:
        for i in range(max_iter):
            self._update()
            self.idx = i
            if self.converge:
                print(' - Converged!')
                break
            # metric computation and early-stopping check
            if self.idx % self.metric_call_period == 0:
                kwargs = self.get_notify_observers_kwargs()
                self.notify_observers('cv_metrics', **kwargs)
                if self.any_convergence_flag():
                    if self.verbose:
                        print("\n-----> early-stopping done")
                    break
            bar.update(self.idx)
    # retrieve metrics results
    self.retrieve_outputs()
    # rename outputs as attributes
    self.x_final = self._z_new


@monkeypatch(Condat)
def iterate(self, max_iter=150):
    r"""Iterate

    This method calls update until either convergence criteria is met or
    the maximum number of iterations is reached

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of iterations (default is ``150``)

    """

    with progressbar.ProgressBar(redirect_stdout=True,
                                 max_value=max_iter) as bar:

        for i in range(max_iter):
            self._update()
            self.idx = i

            if self.converge:
                print(' - Converged!')
                break
            # metric computation and early-stopping check
            if self.idx % self.metric_call_period == 0:
                kwargs = self.get_notify_observers_kwargs()
                self.notify_observers('cv_metrics', **kwargs)
                if self.any_convergence_flag():
                    if self.verbose:
                        print("\n-----> early-stopping done")
                    break
            bar.update(self.idx)

    # retrieve metrics results
    self.retrieve_outputs()

    # rename outputs as attributes
    self.x_final = self._x_new
    self.y_final = self._y_new
