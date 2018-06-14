"""
Neuroimaging non-cartesian Parallel reception reconstruction
============================================================

Credit: L Elgueddari, S.Lannuzel

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

"""

# Package import
import pysap
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct.reconstruct import NFFT2
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.parallel_mri.gradient import Gradient_pMRI

# Third party import
import numpy as np
import matplotlib.pyplot as plt

# Loading input data
Il = get_sample_data("2d-pmri").data
SOS = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))
Smaps = np.asarray([Il[channel]/SOS for channel in range(Il.shape[0])])

samples = get_sample_data("mri-radial-samples").data

plt.imshow(np.abs(SOS), cmap='gray')
plt.axis('off')
plt.show()

plt.plot(samples[:, 0], samples[:, 1], '.k')
plt.axis('off')
plt.show()

#############################################################################
# Generate the kspace
# -------------------

# Get the locations of the kspace samples and the associated observations

fourier_op_c = NFFT2(samples=samples, shape=SOS.shape)

# Generate the subsampled kspace
Sl = [Smaps[channel] * SOS for channel in range(Smaps.shape[0])]

kspace_data = np.asarray([fourier_op_c.op(I) for I in Sl])
Sl = np.asarray(Sl)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 10

linear_op = Wavelet2(wavelet_name="UndecimatedBiOrthogonalTransform",
                     nb_scale=4)

fourier_op = NFFT2(samples=samples, shape=SOS.shape)

gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op,
                            S=Smaps)

x_final, transform, cost = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    mu=1e-9,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1,
    get_cost=True)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()

plt.figure()
plt.plot(cost)
plt.show()
#############################################################################
# Condata-Vu optimization
# -----------------------
#
# We now want to refine the zero order solution using a Condata-Vu
# optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the CONDAT-VU reconstruction
max_iter = 10
gradient_op_cd = Gradient_pMRI(data=kspace_data,
                               fourier_op=fourier_op,
                               S=Smaps)

x_final, transform = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    std_est=None,
    std_est_method=None,
    std_thr=2.,
    mu=0,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=0,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)

image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
