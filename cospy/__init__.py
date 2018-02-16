##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
COSPY is a Python package related to sparsity and its application in
astronomical data analysis. This package is based on the 'sparse2d' C++ library
that allows sparse decomposition, denoising and deconvolution.
"""

# import matplotlib
# matplotlib.use("Agg")

from __future__ import print_function
from .info import __version__
import cospy.extensions
from cospy.base import io
from cospy.utils import TempDir
from cospy.configure import info
from cospy.base.image import Image
from cospy.utils import load_transform
from cospy.base.utils import monkeypatch
from cospy.utils import AVAILABLE_TRANSFORMS


# Apply some monkeypatchs to the optimization package
import progressbar
from modopt.opt.algorithms import Condat
from modopt.opt.algorithms import ForwardBackward


# Display a welcome message
print(info())


@monkeypatch(ForwardBackward)
def iterate(self, max_iter=150):
    """ Monkey patch the optimizer iterate method to have a progressbar.
    """
    with progressbar.ProgressBar(redirect_stdout=True,
                                 max_value=max_iter) as bar:
        for idx in range(max_iter):
            self._update()
            if self.converge:
                print(' - Converged!')
                break
            bar.update(idx)
    self.x_final = self._z_new


@monkeypatch(Condat)
def iterate(self, max_iter=150):
    """ Monkey patch the optimizer iterate method to have a progressbar.
    """
    with progressbar.ProgressBar(redirect_stdout=True,
                                 max_value=max_iter) as bar:
        for idx in range(max_iter):
            self._update()
            if self.converge:
                print(' - Converged!')
                break
            bar.update(idx)
    self.x_final = self._x_new
    self.y_final = self._y_new
