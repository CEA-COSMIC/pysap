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
The Condat-Vu splitting method is a primal-dual proximal algorithm.

**References**

1) Condat, A Primal-Dual Splitting Method for Convex Optimization Involving
Lipschitzian, Proximable and Linear Composite Terms, 2013, Journal of
Optimization Theory and Applications, 158, 2, 460. (C2013)
2) Bauschke et al., Fixed-Point Algorithms for Inverse Problems in Science
and Engineering, 2011, Chapter 10. (B2010)
3) Raguet et al., Generalized Forward-Backward Splitting, 2012, , (R2012)

**Notes**

* x_old is used in place of x_{n}.
* x_new is used in place of x_{n+1}.
* x_prox is used in place of \~{x}_{n+1}.
* x_temp is used for intermediate operations.
"""

# System import
from __future__ import division, print_function, absolute_import
import numpy as np
import copy


class FISTA():
    """ FISTA

    This class is inhereited by optimisation classes to speed up convergence

    Parameters
    ----------
    lambda_init : float
        Initial value of the relaxation parameter
    active : bool
        Option to activate FISTA convergence speed-up (default is 'True')
    """

    def __init__(self, lambda_init=None, active=True):

        self.lambda_now = lambda_init
        self.t_now = 1.0
        self.t_prev = 1.0
        self.use_speed_up = active

    def speed_switch(self, turn_on=True):
        """ Speed swicth

        This method turns on or off the speed-up

        Parameters
        ----------
        turn_on : bool
            Option to turn on speed-up (default is 'True')

        """
        self.use_speed_up = turn_on

    def update_lambda(self):
        """ Update lambda

        This method updates the value of lambda

        Notes
        -----
        Implements steps 3 and 4 from algoritm 10.7 in B2010

        """
        self.t_prev = self.t_now
        self.t_now = (1 + np.sqrt(4 * self.t_prev ** 2 + 1)) * 0.5
        self.lambda_now = 1 + (self.t_prev - 1) / self.t_now

    def speed_up(self):
        """ Speed-up

        This method returns the update if the speed-up is active

        """
        if self.use_speed_up:
            self.update_lambda()


class ForwardBackward(FISTA):
    """ Forward-Backward optimisation

    This class implements standard forward-backward optimisation with an the
    option to use the FISTA speed-up

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    grad : class
        Gradient operator class
    prox : class
        Proximity operator class
    cost : class
        Cost function class
    lambda_init : float
        Initial value of the relaxation parameter
    lambda_update :
        Relaxation parameter update method
    use_fista : bool
        Option to use FISTA (default is 'True')
    auto_iterate : bool
        Option to automatically begin iterations upon initialisation (default
        is 'True')
    """

    def __init__(self, x, grad, prox, cost=None, lambda_init=None,
                 lambda_update=None, use_fista=True, auto_iterate=True):
        FISTA.__init__(self, lambda_init, use_fista)
        self.x_old = x
        self.z_old = np.copy(self.x_old)
        self.grad = grad
        self.prox = prox
        self.cost_func = cost
        self.lambda_update = lambda_update
        self.converge = False
        if auto_iterate:
            self.iterate()

    def update(self):
        """ Update

        This method updates the current reconstruction

        Notes
        -----
        Implements algorithm 10.7 (or 10.5) from B2010
        """
        # Step 1 from alg.10.7.
        self.grad.get_grad(self.z_old)
        y_old = self.z_old - self.grad.inv_spec_rad * self.grad.grad

        # Step 2 from alg.10.7.
        self.x_new = self.prox.op(y_old)

        # Steps 3 and 4 from alg.10.7.
        self.speed_up()

        # Step 5 from alg.10.7.
        self.z_new = self.x_old + self.lambda_now * (self.x_new - self.x_old)

        # Test primal variable for convergence.
        if np.sum(np.abs(self.z_old - self.z_new)) <= 1e-6:
            print(' - converged!')
            self.converge = True

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)
        np.copyto(self.z_old, self.z_new)

        # Update parameter values for next iteration.
        if not isinstance(self.lambda_update, type(None)):
            self.lambda_now = self.lambda_update(self.lambda_now)

        # Test cost function for convergence.
        if not isinstance(self.cost_func, type(None)):
            self.converge = self.cost_func.get_cost(self.z_new)

        if np.all(self.z_new == 0.0):
            raise RuntimeError('The reconstruction is fucked!')

    def iterate(self, max_iter=150):
        """ Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is '150')
        """
        for i in xrange(max_iter):
            self.update()

            if self.converge:
                print(' - Converged!')
                break

        self.x_final = self.z_new


class GenForwardBackward():
    """ Generalized Forward-Backward optimisation

    This class implements algorithm 1 from R2012

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    grad : class
        Gradient operator class
    prox_list : list
        List of proximity operator classes
    cost : class
        Cost function class
    lambda_init : float
        Initial value of the relaxation parameter
    lambda_update :
        Relaxation parameter update method
    weights : np.ndarray
        Proximity operator weights
    auto_iterate : bool
        Option to automatically begin iterations upon initialisation (default
        is 'True')
    """

    def __init__(self, x, grad, prox_list, cost=None, lambda_init=1.0,
                 lambda_update=None, weights=None, auto_iterate=True,
                 plot=False):
        self.x_old = x
        self.grad = grad
        self.prox_list = np.array(prox_list)
        self.cost_func = cost
        self.lambda_init = lambda_init
        self.lambda_update = lambda_update

        if isinstance(weights, type(None)):
            self.weights = np.repeat(1.0 / self.prox_list.size,
                                     self.prox_list.size)
        else:
            self.weights = np.array(weights)

        # Check weights.
        if np.sum(self.weights) != 1.0:
            raise ValueError('Proximity operator weights must sum to 1.0.'
                             'Current sum of weights = ' +
                             str(np.sum(self.weights)))

        self.z = np.array([self.x_old for i in xrange(self.prox_list.size)])

        self.plot = plot
        self.converge = False
        if auto_iterate:
            self.iterate()

    def update(self):
        """ Update

        This method updates the current reconstruction

        Notes
        -----
        Implements algorithm 1 from R2012
        """
        # Calculate gradient for current iteration.
        self.grad.get_grad(self.x_old)

        # Update z values.
        for i in xrange(self.prox_list.size):
            z_temp = (2 * self.x_old - self.z[i] - self.grad.inv_spec_rad *
                      self.grad.grad)
            z_prox = self.prox_list[i].op(z_temp,
                                          extra_factor=self.grad.inv_spec_rad /
                                          self.weights[i])
            self.z[i] += self.lambda_init * (z_prox - self.x_old)

        # Update current reconstruction.
        self.x_new = np.sum((z_i * w_i for z_i, w_i in
                            zip(self.z, self.weights)), axis=0)

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)

        # Update parameter values for next iteration.
        if not isinstance(self.lambda_update, type(None)):
            self.lambda_now = self.lambda_update(self.lambda_now)

        # Test cost function for convergence.
        if not isinstance(self.cost_func, type(None)):
            self.converge = self.cost_func.get_cost(self.x_new)

        if np.all(self.x_new == 0.0):
            raise RuntimeError('The deconvolution is fucked!')

    def iterate(self, max_iter=150):
        """ Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is '150')
        """
        for i in xrange(max_iter):
            self.update()

            if self.converge:
                print(' - Converged!')
                break

        self.x_final = self.x_new
        # self.cost_func.plot_cost()


class Condat():
    """ Condat optimisation

    This class implements algorithm 10.7 from C2013

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    y : np.ndarray
        Initial guess for the dual variable
    grad : class
        Gradient operator class
    prox : class
        Proximity primal operator class
    prox_dual : class
        Proximity dual operator class
    linear : class
        Linear operator class
    cost : class
        Cost function class
    rho : float
        Relaxation parameter
    sigma : float
        Proximal dual parameter
    tau : float
        Proximal primal paramater
    rho_update :
        Relaxation parameter update method
    sigma_update :
        Proximal dual parameter update method
    tau_update :
        Proximal primal parameter update method
    extra_factor_update :
        Extra factor passed to the dual proximity operator update
    auto_iterate : bool
        Option to automatically begin iterations upon initialisation (default
        is 'True')
    """

    def __init__(self, x, y, grad, prox, prox_dual, linear, cost,
                 rho,  sigma, tau, rho_update=None, sigma_update=None,
                 tau_update=None, extra_factor_update=None, auto_iterate=True):
        self.x_old = x
        self.y_old = y
        self.grad = grad
        self.prox = prox
        self.prox_dual = prox_dual
        self.linear = linear
        self.cost_func = cost
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.rho_update = rho_update
        self.sigma_update = sigma_update
        self.tau_update = tau_update
        self.converge = False
        self.extra_factor = 1.
        self.extra_factor_update = extra_factor_update
        if auto_iterate:
            self.iterate()

    def update_param(self):
        """ Update parameters

        This method updates the values of rho, sigma and tau with the methods
        provided
        """
        # Update relaxation parameter.
        if self.rho_update is not None:
            self.rho = self.rho_update(self.rho)

        # Update proximal dual parameter.
        if self.sigma_update is not None:
            self.sigma = self.sigma_update(self.sigma)

        # Update proximal primal parameter.
        if self.tau_update is not None:
            self.tau = self.tau_update(self.tau)

        # Update the dual proximity extra factor
        if self.extra_factor_update is not None:
            self.extra_factor = self.extra_factor_update(
                self.grad, self.linear)

    def update(self):
        """ Update

        This method updates the current reconstruction

        Notes
        -----
        Implements equation 9 (algorithm 3.1) from C2013
        """
        # Update parameter values for next iteration.
        self.grad.get_grad(self.x_old)
        self.update_param()

        # Step 1 from eq.9.
        x_temp = (self.x_old - self.tau * self.grad.grad - self.tau *
                  self.linear.adj_op(self.y_old))
        x_prox = self.prox.op(x_temp)

        # Step 2 from eq.9.
        y_temp = (self.y_old + self.sigma *
                  self.linear.op(2 * x_prox - self.x_old))
        y_prox = (y_temp - self.sigma *
                  self.prox_dual.op(y_temp, extra_factor=self.extra_factor))

        # Step 3 from eq.9.
        self.x_new = self.rho * x_prox + (1 - self.rho) * self.x_old
        self.y_new = self.rho * y_prox + (1 - self.rho) * self.y_old

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)
        y_old = copy.deepcopy(self.y_new)

        # Test cost function for convergence.
        self.converge = self.cost_func.get_cost(self.x_new)

    def iterate(self, max_iter=150):
        """ Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is '150')
        """

        for i in range(max_iter):
            self.update()

            if self.converge:
                print(" - Converged!")
                break

        self.x_final = self.x_new
        self.y_final = self.y_new
        # self.cost_func.plot_cost()
