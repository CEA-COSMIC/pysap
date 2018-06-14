# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Noise estimation strategies.
"""


# Package import
from .utils import flatten

# Third party import
from modopt.math.stats import sigma_mad


def sigma_mad_sparse(grad_op, linear_op):
    """ Estimate the std from the mad routine on each approximation scale.

    Parameters
    ----------
    grad_op: instance
        gradient operator.
    linear_op: instance
        linear operator.

    Returns
    -------
    sigma: list of float
        a list of std estimate for each scale.
    """
    linear_op.op(grad_op.grad)
    return [sigma_mad(flatten(linear_op.transform[scale])[0])
            for scale in range(linear_op.transform.nb_scale)]
