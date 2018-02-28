# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains classses for defining algorithm operators and gradients.
"""


# System import
import copy
from builtins import range

# Package import
import pysap
from .utils import flatten

# Third party import
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic
import numpy as np
import scipy.fftpack as pfft


class GradAnalysis2(GradBasic, PowerMethod):

    def __init__(self, data, fourier_op):

        GradBasic.__init__(self, data, fourier_op.op, fourier_op.adj_op)
        self.fourier_op = fourier_op
        PowerMethod.__init__(self, self.trans_op_op, self.fourier_op.shape,
                             data_type=np.complex, auto_run=False)
        self.get_spec_rad(extra_factor=1.1)


class GradSynthesis2(GradBasic, PowerMethod):

    def __init__(self, data, linear_op, fourier_op):

        GradBasic.__init__(self, data, self._op_method, self._trans_op_method)
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        coef = linear_op.op(np.zeros(fourier_op.shape).astype(np.complex))
        PowerMethod.__init__(self, self.trans_op_op, coef.shape,
                             data_type=np.complex, auto_run=False)
        self.get_spec_rad(extra_factor=1.1)

    def _op_method(self, data, *args, **kwargs):

        return self.fourier_op.op(self.linear_op.adj_op(data))

    def _trans_op_method(self, data, *args, **kwargs):

        return self.linear_op.op(self.fourier_op.adj_op(data))
