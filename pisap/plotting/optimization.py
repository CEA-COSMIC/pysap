# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
#
#:Author: Samuel Farrens <samuel.farrens@gmail.com>
#:Version: 1.1
#:Date: 04/01/2017
##########################################################################

"""
This module contains methods for monitoring optimzation routines.
"""

# System import
import numpy as np
import matplotlib.pyplot as plt


def plot_cost(cost_list, out_fname=None):
    """ Plot the final cost function.

    Parameters
    ----------
    cost_list: list
        List of cost function values.
    out_fname: str (optional, default None)
        The output file name.
    """
    plt.figure("Cost Function")
    plt.plot(cost_list, "r-")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    if out_fname is not None:
        plt.savefig(out_fname)
        plt.close()

