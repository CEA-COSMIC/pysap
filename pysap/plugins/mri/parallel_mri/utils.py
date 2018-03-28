# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains all the utils tools needed in the p_MRI reconstruction.
"""


# System import

# Package import

# Third party import
import numpy as np


def prod_over_maps(S, X):
    """
    Computes the element-wise product of the two inputs over the first two
    direction

    Parameters
    ----------
    S: np.ndarray
        The sensitivity maps of size [N,M,L]
    X: np.ndarray
        An image of size [N,M]

    Returns
    -------
    Sl: np.ndarray
        The product of every L element of S times X
    """
    if S.shape == X.shape:
        Sl = [S[l] * X[l] for l in range(len(S))]
    else:
        Sl = [S_coil * X for S_coil in S]
    return np.asarray(Sl)


def function_over_maps(f, x):
    """
    This methods computes the callable function over the third direction

    Parameters
    ----------
    f: callable
        This function will be applyed n times where n is the last element in
        the shape of x
    x: np.ndarray
        Input data

    Returns
    -------
    out: np.list
        the results of the function as a list where the length of the list is
        equal to n
    """
    return np.asarray([f(xl) for xl in x])


def check_lipschitz_cst(f, x_shape, lipschitz_cst, max_nb_of_iter=10):
    """
    This methods check that for random entrees the lipschitz constraint are
    statisfied:

    * ||f(x)-f(y)|| < lipschitz_cst ||x-y||

    Parameters
    ----------
    f: callable
        This lipschitzien function
    x_shape: tuple
        Input data shape
    lipschitz_cst: float
        The Lischitz constant for the function f
    max_nb_of_iter: int
        The number of time the constraint must be satisfied

    Returns
    -------
    out: bool
        If is True than the lipschitz_cst given in argument seems to be an
        upper bound of the real lipschitz constant for the function f
    """
    is_lips_cst = True
    n = 0

    while is_lips_cst and n < max_nb_of_iter:
        n += 1
        x = np.random.randn(*x_shape)
        y = np.random.randn(*x_shape)
        is_lips_cst = (np.linalg.norm(f(x)-f(y)) <= (lipschitz_cst *
                                                     np.linalg.norm(x-y)))

    return is_lips_cst
