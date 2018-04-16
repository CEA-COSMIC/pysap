from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity
from modopt.opt.cost import costObj
from pysap.numerics.linear import Wavelet2
from pysap.numerics.fourier import FFT2, NFFT2
from pysap.numerics.gradient import GradSynthesis2, GradAnalysis2
from pysap.numerics.opt import opt_fb, opt_cv
from pysap.plugins.mri.parallel_mri.gradient import Grad2D_pMRI

import numpy as np


def rec_ista_2d(data, wavelet_name, samples, mu, nb_scale=4, max_iter=150,
                tol=1e-4, cartesian_sampling=True, uniform_data_shape=None,
                acceleration=True, cost=None, **kwargs):
    """
    solve the synthesis problem :
    min 0.5||F Psi* alpha - y||^2 + mu||alpha||_1
    where Psi is the wavelet transform. It should therefore be an orthogonal
    basis. The optimization solver is ISTA or FISTA if acceleration is on
    :param data: np.ndarray
    Data in the Fourier domain
    :param wavelet_name: str
    Wavelet transform name, should be in pysap.AVAILABLE_TRANSFORMS
    :param samples: np.ndarray
    Position of k_space samples
    :param mu: float
    regularization parameter
    :param nb_scale: int, default 4
    number of decomposition scales
    :param max_iter: int, default 150
    Number of iterations
    :param tol: float, default 1e-4
    tolerance of convergence, used if cost is auto
    :param cartesian_sampling: bool
    Specify if sampling is cartesian or not
    :param uniform_data_shape: tuple
    image shape, needed if sampling is not cartesian
    :param acceleration: bool, default True
    Use Nesterov's acceleration in fb algorithm
    :param cost: cost method
    either None, 'auto' or a CostObj instance
    :param kwargs: extra parameters
    :return: x_final: np.ndarray
    Result of the reconstruction, such that x = Psi* alpha
    """
    verbose = kwargs.get('verbose', 1)
    linear_op = Wavelet2(wavelet_name=wavelet_name,
                         nb_scale=nb_scale)
    # TODO check if the wavelet transform is an orthogonal basis,
    # otherwise print a warning

    # Definition of Fourier operator
    if cartesian_sampling:
        fourier_op = FFT2(samples=samples, shape=data.shape)
    elif uniform_data_shape is None:
        raise AttributeError('If using the NUFFT, the uniform_data_shape '
                             'argument should be provided')
    else:
        fourier_op = NFFT2(samples=samples, shape=uniform_data_shape)

    # gradient operator
    gradient_op = GradSynthesis2(data=data,
                                 linear_op=linear_op,
                                 fourier_op=fourier_op)

    # Algorithm initialization
    x_init = kwargs.get('x_init', np.zeros(fourier_op.shape, dtype=np.complex))
    alpha_init = linear_op.op(x_init)
    weights = mu * np.ones(alpha_init.shape)

    # proximity operator associated with the l1 norm, the linear operator
    # is Identity for the penalty is lambda * ||alpha||_1
    prox_op = SparseThreshold(linear=Identity(),
                              weights=weights,
                              thresh_type='soft')

    # Define cost operator, optional
    if cost == 'auto':
        cost = costObj([gradient_op, prox_op],
                       tolerance=tol,
                       verbose=verbose)

    # Use or not Nesterov's acceleration in ISTA algorithm
    if not acceleration:
        def lambda_update(lambda_param_old): return 1.0
    else:
        lambda_update = 'fista'

    alpha_final = opt_fb(gradient_op=gradient_op,
                         prox_op=prox_op,
                         x_init=alpha_init,
                         beta_param=gradient_op.inv_spec_rad,
                         lambda_update=lambda_update,
                         max_iter=max_iter,
                         cost=cost,
                         **kwargs
                         )
    x_final = linear_op.adj_op(alpha_final)
    return x_final


def rec_condat_vu_2d(data, wavelet_name, samples, mu, nb_scale=4,
                     max_iter=150, tol=1e-4, cartesian_sampling=True,
                     uniform_data_shape=None, cost=None, **kwargs):
    """
    Solve the analysis problem :
    min 0.5 ||Fx - y||^2 + lambda * ||Psi x||_1
    using Condat-Vu algorithm
    :param data: np.ndarray
    Data in the Fourier domain
    :param wavelet_name: str
    Wavelet transform name, should be in pysap.AVAILABLE_TRANSFORMS
    :param samples: np.ndarray
    Position of k_space samples
    :param mu: float
    regularization parameter
    :param nb_scale: int, default 4
    number of decomposition scales
    :param max_iter: int, default 150
    Number of iterations
    :param tol: float, default 1e-4
    tolerance of convergence, used if cost is auto
    :param cartesian_sampling: bool
    Specify if sampling is cartesian or not
    :param uniform_data_shape: tuple
    image shape, needed if sampling is not cartesian
    :param cost: cost method
    either None, 'auto' or a CostObj instance
    :param kwargs: extra parameters
    :return: x_final: np.ndarray
    The reconstructed image
    :return: alpha_final: np.ndarray
    The reconstructed dual variable
    """
    # TODO add std estimation and reweightings
    verbose = kwargs.get('verbose', 1)
    linear_op = Wavelet2(wavelet_name=wavelet_name,
                         nb_scale=nb_scale)
    if cartesian_sampling:
        data_shape = data.shape
        fourier_op = FFT2(samples=samples, shape=data_shape)
    elif uniform_data_shape is None:
        raise AttributeError('If using the NUFFT, the uniform_data_shape '
                             'argument should be provided')
    else:
        data_shape = uniform_data_shape
        fourier_op = NFFT2(samples=samples, shape=data_shape)

    gradient_op = GradAnalysis2(data=data, fourier_op=fourier_op)

    x_init = kwargs.get('x_init', np.zeros(fourier_op.shape, dtype=np.complex))
    alpha_init = linear_op.op(x_init)
    weights = mu * np.ones(alpha_init.shape)

    # Define the proximity operator associated with the H function in Condat
    # algorithm, which is here the l1 norm
    prox_op_dual = SparseThreshold(linear=linear_op,
                                   weights=weights,
                                   thresh_type='soft')

    # Define the proximity operator associated with the G function in Condat
    # algorithm, which is here the zero function, hence the prox is Identity
    prox_op = Identity()

    # condat-vu parameters
    sigma = kwargs.get('sigma', 0.5)
    if 'tau' not in kwargs.keys():
        tau = 1.0 / (gradient_op.spec_rad / 2.0 + sigma *
                     linear_op.l2norm(data_shape)**2 + 1e-8)
    else:
        tau = kwargs.get('tau')

    if 1/tau-sigma*linear_op.l2norm(data_shape)**2 < gradient_op.spec_rad/2.0:
        print("WARNING, the parameters tau and sigma do not respect "
              "inequality condition")

    if cost == 'auto':
        cost = costObj([gradient_op, prox_op_dual],
                       tolerance=tol,
                       verbose=verbose)
    elif cost == 'dual':
        raise NotImplementedError('Come back in a few')

    x_final, alpha_final = opt_cv(gradient_op=gradient_op,
                                  prox_op_g=prox_op,
                                  prox_op_h=prox_op_dual,
                                  primal_init=x_init,
                                  dual_init=alpha_init,
                                  max_iter=max_iter,
                                  cost=cost,
                                  tau=tau,
                                  sigma=sigma,
                                  **kwargs)
    return x_final, alpha_final


def rec_ista_2d_p(data, wavelet_name, samples, mu, s_maps, nb_scale=4,
                  max_iter=150, tol=1e-4, cartesian_sampling=True,
                  uniform_data_shape=None, acceleration=True, cost=None,
                  **kwargs):
    """
    solve the synthesis problem :
    min 0.5||F S Psi* alpha - y||^2 + mu||alpha||_1
    where Psi is the wavelet transform. It should therefore be an orthogonal
    basis. The optimization solver is ISTA
    :param data: np.ndarray
    Data in the Fourier domain
    :param wavelet_name: str
    Wavelet transform name, should be in pysap.AVAILABLE_TRANSFORMS
    :param samples: np.ndarray
    Position of k_space samples
    :param mu: float
    regularization parameter
    :param s_maps: np.ndarray
    sensitivity maps
    :param nb_scale: int, default 4
    number of decomposition scales
    :param max_iter: int, default 150
    Number of iterations
    :param tol: float, default 1e-4
    tolerance of convergence, used if cost is auto
    :param cartesian_sampling: bool
    Specify if sampling is cartesian or not
    :param uniform_data_shape: tuple
    image shape, needed if sampling is not cartesian
    :param acceleration: bool, default True
    Use Nesterov's acceleration in fb algorithm
    :param cost: cost method
    either None, 'auto', 'dual' or a CostObj instance
    :param kwargs: extra parameters
    :return: x_final: np.ndarray
    Result of the reconstruction, such that x = Psi* alpha
    """
    verbose = kwargs.get('verbose', 1)
    linear_op = Wavelet2(wavelet_name=wavelet_name,
                         nb_scale=nb_scale)
    # TODO check if the wavelet transform is an orthogonal basis,
    # otherwise print a warning

    # Definition of Fourier operator
    if cartesian_sampling:
        fourier_op = FFT2(samples=samples, shape=data.shape[0:2])
    elif uniform_data_shape is None:
        raise AttributeError('If using the NUFFT, the uniform_data_shape '
                             'argument should be provided')
    else:
        fourier_op = NFFT2(samples=samples, shape=uniform_data_shape)

    # gradient operator
    gradient_op = Grad2D_pMRI(data=data, linear_op=linear_op,
                              fourier_op=fourier_op, S=s_maps)

    # Algorithm initialization
    x_init = kwargs.get('x_init', np.zeros(fourier_op.shape, dtype=np.complex))
    alpha_init = linear_op.op(x_init)
    weights = mu * np.ones(alpha_init.shape)

    # proximity operator associated with the l1 norm, the linear operator
    # is Identity for the penalty is lambda * ||alpha||_1
    prox_op = SparseThreshold(linear=Identity(),
                              weights=weights,
                              thresh_type='soft')

    # Define cost operator, optional
    if cost == 'auto':
        cost = costObj([gradient_op, prox_op], tolerance=tol, verbose=verbose)

    # Use or not Nesterov's acceleration in ISTA algorithm
    if not acceleration:
        def lambda_update(lambda_param_old): return 1.0
    else:
        lambda_update = 'fista'

    alpha_final = opt_fb(gradient_op=gradient_op,
                         prox_op=prox_op,
                         x_init=alpha_init,
                         beta_param=gradient_op.inv_spec_rad,
                         lambda_update=lambda_update,
                         max_iter=max_iter,
                         cost=cost,
                         verbose=verbose,
                         **kwargs)

    x_final = linear_op.adj_op(alpha_final)
    return x_final


def rec_condat_vu_2d_p(data, wavelet_name, samples, mu, s_maps, nb_scale=4,
                       max_iter=150, tol=1e-4, cartesian_sampling=True,
                       uniform_data_shape=None, cost=None, **kwargs):
    """
    Solve the analysis problem :
    min 0.5 ||Fx - y||^2 + lambda * ||Psi x||_1
    using Condat-Vu algorithm

    :param data: np.ndarray
    Data in the Fourier domain
    :param wavelet_name: str
    Wavelet transform name, should be in pysap.AVAILABLE_TRANSFORMS
    :param samples: np.ndarray
    Position of k_space samples
    :param mu: float
    regularization parameter
    :param s_maps: np.ndarray
    sensitivity maps
    :param nb_scale: int, default 4
    number of decomposition scales
    :param max_iter: int, default 150
    Number of iterations
    :param tol: float, default 1e-4
    tolerance of convergence, used if cost is auto
    :param cartesian_sampling: bool
    Specify if sampling is cartesian or not
    :param uniform_data_shape: tuple
    image shape, needed if sampling is not cartesian
    :param cost: cost method
    either None, 'auto' or a CostObj instance
    :param kwargs: extra parameters

    :return: x_final: np.ndarray
    The reconstructed image
    :return: alpha_final: np.ndarray
    The reconstructed dual variable
    """
    # TODO add std estimation and reweightings
    verbose = kwargs.get('verbose', 1)
    linear_op = Wavelet2(wavelet_name=wavelet_name,
                         nb_scale=nb_scale)
    if cartesian_sampling:
        data_shape = data.shape[0:2]
        fourier_op = FFT2(samples=samples, shape=data_shape)
    elif uniform_data_shape is None:
        raise AttributeError('If using the NUFFT, the uniform_data_shape '
                             'argument should be provided')
    else:
        data_shape = uniform_data_shape
        fourier_op = NFFT2(samples=samples, shape=data_shape)

    gradient_op = Grad2D_pMRI(data=data, fourier_op=fourier_op, S=s_maps)
    gradient_op.spec_rad = 1.1
    gradient_op.inv_spec_rad = 1.0/1.1

    x_init = kwargs.get('x_init', np.zeros(fourier_op.shape, dtype=np.complex))
    alpha_init = linear_op.op(x_init)
    weights = mu * np.ones(alpha_init.shape)

    # Define the proximity operator associated with the H function in Condat
    # algorithm, which is here the l1 norm
    prox_op_dual = SparseThreshold(linear=linear_op,
                                   weights=weights,
                                   thresh_type='soft')

    # Define the proximity operator associated with the G function in Condat
    # algorithm, which is here the zero function, hence the prox is Identity
    prox_op = Identity()

    # condat-vu parameters
    sigma = kwargs.get('sigma', 0.5)
    if 'tau' not in kwargs.keys():
        tau = 1.0 / (gradient_op.spec_rad / 2.0 + sigma *
                     linear_op.l2norm(data_shape)**2 + 1e-8)
    else:
        tau = kwargs.get('tau')

    if 1/tau-sigma*linear_op.l2norm(data_shape)**2 < gradient_op.spec_rad/2.0:
        print("WARNING, the parameters tau and sigma do not respect "
              "inequality condition")
    rho = kwargs.get('rho', 1.0)

    if cost == 'auto':
        cost = costObj([gradient_op, prox_op_dual],
                       tolerance=tol,
                       verbose=verbose)
    elif cost == 'dual':
        raise NotImplementedError('Come back in a few')

    x_final, alpha_final = opt_cv(gradient_op=gradient_op,
                                  prox_op_g=prox_op,
                                  prox_op_h=prox_op_dual,
                                  primal_init=x_init,
                                  dual_init=alpha_init,
                                  max_iter=max_iter,
                                  cost=cost,
                                  tau=tau,
                                  sigma=sigma,
                                  rho=rho,
                                  **kwargs)
    return x_final, alpha_final
