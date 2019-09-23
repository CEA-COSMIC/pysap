# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
#import os
import warnings

# Package import
import pysap
from pysap.base.transform import MetaRegister # for the metaclass

import matplotlib.pyplot as plt
import numpy as np

try:
    import pysparse
except ImportError:
    warnings.warn("Sparse2d python bindings not found, use binaries.")
    pysparse = None

# Third party import
import numpy

class Filter():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data = None

    def filter(self, data):
        flt = pysparse.MRFilters(**(self.kwargs))
        self.data = flt.filter(data)

    def show(self, save=False):
        if self.data is None:
            raise Exception("Data needs to be filtered befor it can be shown !")
        plt.imshow(self.data, cmap='gray')
        plt.suptitle("Filtered Image")
        if save:
            plt.savefig()
        plt.imshow(self.data, cmap='gray', vmax = np.max(self.data), vmin=0)
        plt.suptitle("Filtered Image")
        plt.show()
