##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
piSAP is a Python package related to sparsity and its application in
astronomical data analysis. This package is based on the 'sparse2d' C++ library
that allows sparse decomposition, denoising and deconvolution.
"""

# import matplotlib
# matplotlib.use("Agg")

from .info import __version__
import pisap.extensions
from pisap.base import io
from pisap.utils import TempDir
from pisap.base.image import Image
from pisap.utils import load_transform
from pisap.base.utils import monkeypatch
from pisap.utils import AVAILABLE_TRANSFORMS


# Apply some monkeypatchs to the optimization package
import progressbar
from sf_tools.signal.optimisation import Condat
from sf_tools.signal.optimisation import ForwardBackward


@monkeypatch(ForwardBackward)
def iterate(self, max_iter=150):
    """ Monkey patch the optimizer iterate method to have a progressbar.
    """
    with progressbar.ProgressBar(redirect_stdout=True,
                                 max_value=max_iter) as bar:
        for idx in range(max_iter):
            self.update()
            if self.converge:
                print(' - Converged!')
                break
            bar.update(idx)
    self.x_final = self.z_new


@monkeypatch(Condat)
def iterate(self, max_iter=150):
    """ Monkey patch the optimizer iterate method to have a progressbar.
    """
    with progressbar.ProgressBar(redirect_stdout=True,
                                 max_value=max_iter) as bar:
        for idx in range(max_iter):
            self.update()
            if self.converge:
                print(' - Converged!')
                break
            bar.update(idx)
    self.x_final = self.x_new
    self.y_final = self.y_new
