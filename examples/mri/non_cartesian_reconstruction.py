"""
Neuroimaging non-cartesian reconstruction
=========================================

Credit: A Grigis, L Elgueddari, H Carrie

In this tutorial we will reconstruct an MRI image from non-cartesian kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from pysap.numerics.fourier import NFFT
from pysap.numerics.reconstruct import sparse_rec_fista
from pysap.numerics.reconstruct import sparse_rec_condatvu
from pysap.numerics.utils import generate_operators
from pysap.numerics.utils import convert_mask_to_locations

# Third party import
import numpy as np
import scipy.fftpack as pfft


# Loading input data
image = get_sample_data("mri-slice-nifti")
image.data += np.random.randn(*image.shape) * 20.
mask = get_sample_data("mri-mask")
image.show()
mask.show()


#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Get the locations of the kspace samples and the associated observations
kspace_loc = convert_mask_to_locations(mask.data)
fourier_op = NFFT(samples=kspace_loc, shape=image.shape)
kspace_obs = fourier_op.op(image.data)

# Zero order solution
image_rec0 = pysap.Image(data=fourier_op.adj_op(kspace_obs),
                         metadata=image.metadata)
image_rec0.show()


#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Generate operators
gradient_op, linear_op, prox_op, cost_op = generate_operators(
    data=kspace_obs,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    nb_scales=4,
    non_cartesian=True,
    uniform_data_shape=image.shape,
    gradient_space="synthesis")

# Start the FISTA reconstruction
max_iter = 20
x_final, transform, costs, metrics = sparse_rec_fista(
    gradient_op,
    linear_op,
    prox_op,
    None,
    mu=1e-9,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()


#############################################################################
# Condata-Vu optimization
# -----------------------
#
# We now want to refine the zero order solution using a Condata-Vu
# optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Generate operators
gradient_op, linear_op, prox_op, cost_op = generate_operators(
    data=kspace_obs,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    nb_scales=4,
    non_cartesian=True,
    uniform_data_shape=image.shape,
    gradient_space="analysis")

# Start the CONDAT-VU reconstruction
max_iter = 20
x_final, transform, costs, metrics = sparse_rec_condatvu(
    gradient_op,
    linear_op,
    prox_op,
    cost_op,
    std_est=None,
    std_est_method="dual",
    std_thr=2.,
    mu=1e-9,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=2,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
cost_rec = pysap.Image(data=costs)
cost_rec.show()
