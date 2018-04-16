import time
from modopt.opt.algorithms import Condat, ForwardBackward
from modopt.opt.cost import costObj
from modopt.opt.linear import Identity
from pysap.utils import fista_logo, condatvu_logo


def opt_fb(gradient_op, prox_op, **kwargs):
    # FIXME find zero initialization with proper size if no x_init given
    x_init = kwargs.get('x_init')
    beta_param = kwargs.get('beta_param', 1.0)
    lambda_param = kwargs.get('lambda_param', 1.0)
    beta_update = kwargs.get('beta_update', None)
    lambda_update = kwargs.get('lambda_update', 'fista')
    cost = kwargs.get('cost', None)
    verbose = kwargs.get('verbose', 1)
    max_iter = kwargs.get('max_iter', 150)

    if verbose:
        print(fista_logo())
        print(" - Lipschitz constant: ", gradient_op.spec_rad)
        print(" - data: ", gradient_op.obs_data.shape)
        print(" - max iterations: ", max_iter)
        print(" - iterate variable shape: ", x_init.shape)
        print("-" * 40)

    start = time.clock()

    opt = ForwardBackward(
        x=x_init,
        grad=gradient_op,
        prox=prox_op,
        cost=cost,
        beta_param=beta_param,
        lambda_param=lambda_param,
        beta_update=beta_update,
        lambda_update=lambda_update,
        auto_iterate=False)
    opt.iterate(max_iter)
    end = time.clock()
    if verbose:
        # print(" - final iteration number: ", opt._iteration)
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)
    return opt.x_final


def opt_cv(gradient_op, prox_op_g, prox_op_h, **kwargs):
    primal_init = kwargs.get('primal_init')
    dual_init = kwargs.get('dual_init')
    linear_op = kwargs.get('linear_op', prox_op_h._linear)
    cost = kwargs.get('cost', None)
    relaxation_factor = kwargs.get('rho', 1.0)
    sigma = kwargs.get('sigma', 1.0)
    tau = kwargs.get('tau', 1.0)
    rho_update = kwargs.get('rho_update', None)
    sigma_update = kwargs.get('sigma_update', None)
    tau_update = kwargs.get('tau_update', None)
    max_iter = kwargs.get('max_iter', 150)
    verbose = kwargs.get('verbose', 0)

    if verbose:
        print(condatvu_logo())
        print(" - Lipschitz constant: ", gradient_op.spec_rad)
        print(" - data: ", gradient_op.obs_data.shape)
        print(" - max iterations: ", max_iter)
        print(" - primal variable shape: ", primal_init.shape)
        print(" - dual variable shape: ", dual_init.shape)
        print("-" * 40)

    start = time.clock()

    opt = Condat(
        x=primal_init,
        y=dual_init,
        grad=gradient_op,
        prox=prox_op_g,
        prox_dual=prox_op_h,
        linear=linear_op,
        cost=cost,
        rho=relaxation_factor,
        sigma=sigma,
        tau=tau,
        rho_update=rho_update,
        sigma_update=sigma_update,
        tau_update=tau_update,
        auto_iterate=False)
    opt.iterate(max_iter)

    end = time.clock()
    if verbose:
        # print(" - final iteration number: ", opt._iteration)
        print("Execution time: ", end - start, " seconds")

    return opt.x_final, opt.y_final
