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
#:Date: 05/01/2017
##########################################################################

"""
This module contains classes for defining basic algorithms.
"""

# System import
from __future__ import print_function
import numpy as np
from scipy.linalg import norm


class PowerMethod():
    """ Power method class

    This method performs implements power method to calculate the spectral
    radius of the input data

    Parameters
    ----------
    operator : class
        Operator class instance
    data_shape : tuple
        Shape of the data array
    auto_run : bool
        Option to automatically calcualte the spectral radius upon
        initialisation
    """

    def __init__(self, operator, data_shape, auto_run=True):
        self.op = operator
        self.data_shape = data_shape
        if auto_run:
            self.get_spec_rad()

    def get_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values
        """
        return np.random.random(self.data_shape).astype(np.complex)

    def get_spec_rad(self, tolerance=1e-6, max_iter=150, verbose=0):
        """ Get spectral radius.

        This method calculates the spectral radius.

        Parameters
        ----------
        tolerance : float (optional, default 1e-6)
            Tolerance threshold for convergence.
        max_iter : int (optional, default 150)
            Maximum number of iterations.
        verbose: int (optional, default 0)
            The verbosity level.
        """

        # Set (or reset) values of x.
        x_old = self.get_initial_x()

        # Iterate until the L2 norm of x converges.
        for i in xrange(max_iter):

            x_new = self.op(x_old) / norm(x_old)

            if(np.abs(norm(x_new) - norm(x_old)) < tolerance):
                if verbose > 0:
                    print(" - Power Method converged after %d iterations!" %
                           (i + 1))
                break

            elif i == max_iter - 1:
                print(" - Power Method did not converge after %d "
                      "iterations!" % max_iter)

            np.copyto(x_old, x_new)

        self.spec_rad = norm(x_new)
        self.inv_spec_rad = 1.0 / self.spec_rad

