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

# Package import
import pisap
from .utils import flatten

# Third party import
from sf_tools.signal.gradient import GradBasic
import numpy as np
import scipy.fftpack as pfft


class GradBase(GradBasic):
    """ Basic gradient class.

    This class defines the basic methods that will be defined by derived
    gradient classes.
    """
    def get_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values.
        """
        raise NotImplementedError("'get_initial_x' is an abstract method.")

    def MX(self, x):
        """ MX operation.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data.

        Parameters
        ----------
        x : np.ndarray
            input data array (the recovered image).

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        raise NotImplementedError("'MX' is an abstract method.")

    def MtX(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case inverse fourier transform of the input data

        Parameters
        ----------
        x : np.ndarray
            input data array (the recovered image).

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        raise NotImplementedError("'MtX' is an abstract method.")

    def get_spec_rad(self, tolerance=1e-4, max_iter=20, coef_mul=1.1):
        """ Get the spectral radius.

        Parameters
        ----------
        tolerance: float, default 1e-8
            tolerance threshold for convergence.
        max_iter: int, default 150
            maximum number of iterations.
        verbose: int, default 0
            the verbosity level.
        """
        # Set (or reset) values of x.
        x_old = self.get_initial_x()

        # Iterate until the L2 norm of x converges.
        for i in xrange(max_iter):
            x_old_norm = np.linalg.norm(x_old)
            x_new = self.MtMX(x_old) / x_old_norm
            x_new_norm = np.linalg.norm(x_new)
            if(np.abs(x_new_norm - x_old_norm) < tolerance):
                break
            x_old = copy.deepcopy(x_new)
        self.spec_rad = coef_mul * x_new_norm
        self.inv_spec_rad = 1.0 / self.spec_rad


class GradAnalysis2(GradBase):
    """ Gradient 2D analysis class.
    """
    def __init__(self, data, fourier_op):
        """ Initilize the 'GradAnalysis2' class.

        Parameters
        ----------
        data: np.ndarray
            input 2D data array.
        fourier_op: instance
            a Fourier operator instance derived from the FourierBase' class.
        """
        self.y = data
        self.fourier_op = fourier_op
        self.get_spec_rad()

    def get_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an array of random values.
        """
        return np.random.random(self.fourier_op.shape).astype(np.complex)

    def MX(self, x):
        """ MX operation.

        This method calculates the action of the matrix M on the data X, where
        M is the Fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input 2D data array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        return self.fourier_op.op(x)

    def MtX(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, where M is the inverse fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input 2D data array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        return self.fourier_op.adj_op(x)


class GradSynthesis2(GradBase):
    """ Gradient 2D synthesis class.

    This class defines the grad operators for |M*F*invL*alpha - data|**2.

    Parameters
    ----------
    data: np.ndarray
        input 2D data array.
    linear_op: object
        a linear operator instance.
    fourier_op: instance
        a Fourier operator instance.
    """
    def __init__(self, data, linear_op, fourier_op):
        """ Initilize the 'GradSynthesis2' class.
        """
        self.y = data
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        self.get_spec_rad()

    def get_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values.
        """
        fake_data = np.zeros(self.fourier_op.shape).astype(np.complex)
        coef = self.linear_op.op(fake_data)
        return np.random.random(coef.shape).astype(np.complex)

    def MX(self, x):
        """ MX operation.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x: a WaveletTransformBase derived object
            the analysis coefficients.

        Returns
        -------
        result: np.ndarray
            the operation result (the recovered kspace).
        """
        return self.fourier_op.op(self.linear_op.adj_op(x))

    def MtX(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, where M is the inverse fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input kspace 2D array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        return self.linear_op.op(self.fourier_op.adj_op(x))
