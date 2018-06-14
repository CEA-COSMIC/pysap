"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari, S.Lannuzel

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

"""

# Package import
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct_3D.fourier import FFT3
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.parallel_mri.gradient import Gradient_pMRI
from pysap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pysap.plugins.mri.reconstruct_3D.utils import convert_locations_to_mask_3D

# Third party import
import numpy as np
import scipy.fftpack as pfft
import matplotlib.pyplot as plt

# Loading input data
Il = get_sample_data("3d-pmri")

Iref = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))
Smaps = np.asarray([Il[channel]/Iref for channel in range(Il.shape[0])])

imshow3D(Iref, display=True)

samples = get_sample_data("mri-radial-3d-samples").data

samples = normalize_samples(samples)

cartesian_samples = convert_locations_to_mask_3D(samples, Iref.shape)
imshow3D(cartesian_samples, display=True)

cartesian_samples = pfft.fftshift(cartesian_samples)
#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace

fourier_op = FFT3(samples=convert_mask_to_locations_3D(cartesian_samples),
                  shape=Iref.shape)

kspace_data = np.asarray([fourier_op.op(Il[channel]) for channel
                          in range(Il.shape[0])])


#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 4
linear_op = pyWavelet3(wavelet_name="sym4",
                       nb_scale=4)

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

imshow3D(np.abs(x_final), display=True)

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
max_iter = 1
gradient_op_cd = Gradient_pMRI(data=kspace_data,
                               fourier_op=fourier_op,
                               S=Smaps)
x_final, transform = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    std_est=None,
    std_est_method="dual",
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

imshow3D(np.abs(x_final), display=True)
