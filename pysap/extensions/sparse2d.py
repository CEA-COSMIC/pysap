# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import

import warnings

# Package import
import pysap

from pysap.base.transform import MetaRegister  # for the metaclass

try:
    import pysparse
except ImportError:
    warnings.warn("Sparse2d python bindings not found, use binaries.")
    pysparse = None

# Third party import
import numpy as np
import matplotlib.pyplot as plt


class Filter():
    def __init__(self, **kwargs):
        self.data = None

    def filter(self, data, **kwargs):
        flt = pysparse.MRFilters(**kwargs)
        self.data = flt.filter(data)

    def show(self, save=False):
        if self.data is None:
            raise Exception("Data needs to be filtered!")
        plt.imshow(self.data, cmap='gray')
        plt.suptitle("Filtered Image")
        if save:
            plt.savefig()
        plt.imshow(self.data, cmap='gray', vmax=np.max(self.data), vmin=0)
        plt.suptitle("Filtered Image")
        plt.show()
