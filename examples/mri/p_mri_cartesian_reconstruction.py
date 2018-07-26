"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari, S.Lannuzel

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from pysap.numerics.linear import Wavelet2
from pysap.numerics.fourier import FFT2
from pysap.numerics.reconstruct import sparse_rec_fista
from pysap.numerics.reconstruct import sparse_rec_condatvu
from pysap.numerics.utils import convert_mask_to_locations
from pysap.numerics.utils import convert_locations_to_mask
from pysap.numerics.gradient import Gradient_pMRI
from pysap.numerics.proximity import Threshold

# Third party import
import numpy as np
import scipy.fftpack as pfft


# Loading input data
Il = get_sample_data("2d-pmri").data.astype("complex128")
SOS = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))
Smaps = np.asarray([Il[channel]/SOS for channel in range(Il.shape[0])])
samples = get_sample_data("mri-radial-samples").data
mask = pfft.fftshift(convert_locations_to_mask(samples, SOS.shape))
image = pysap.Image(data=np.abs(SOS))
image.show()


#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
Sl = np.asarray([Smaps[l] * SOS for l in range(Smaps.shape[0])])
kspace_data = np.asarray([mask * pfft.fft2(Sl[l]) for l in
                          range(Sl.shape[0])])
mask = pysap.Image(data=pfft.fftshift(mask))
mask.show()


# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(pfft.fftshift(mask.data))


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
prox_op = Threshold(None)
fourier_op = FFT2(samples=kspace_loc, shape=SOS.shape)
gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op,
                            S=Smaps)

x_final, transform, cost, metrics = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=None,
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

# Start the CONDAT-VU reconstruction
max_iter = 1
gradient_op_cd = Gradient_pMRI(data=kspace_data,
                               fourier_op=fourier_op,
                               S=Smaps)
x_final, transform, cost, metrics = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    prox_dual_op=prox_op,
    cost_op=None,
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
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
