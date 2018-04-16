"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
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
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct.reconstruct import FFT2
from pysap.plugins.mri.parallel_mri.utils import prod_over_maps
from pysap.plugins.mri.parallel_mri.utils import function_over_maps
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations
from pysap.plugins.mri.parallel_mri.gradient import Grad2D_pMRI
from pysap.plugins.mri.parallel_mri.extract_sensitivity_maps import get_Smaps
from pysap.plugins.mri.opt import rec_ista_2d_p, rec_condat_vu_2d_p

# Third party import
import numpy as np
import scipy.fftpack as pfft
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Loading input data
image_name = 'path/to/pmri/image'
k_space_ref = loadmat(image_name)['ref']
k_space_ref /= np.linalg.norm(k_space_ref)
Smaps, SOS = get_Smaps(k_space_ref, (512, 512), mode='FFT')
mask = get_sample_data("mri-mask")
# mask.show()
image = pysap.Image(data=np.abs(SOS), metadata=mask.metadata)
# image.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
Sl = prod_over_maps(Smaps, SOS)
kspace_data = function_over_maps(pfft.fft2, Sl)
kspace_data = prod_over_maps(kspace_data, mask.data)
mask.show()

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(mask.data)


#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 10
x_final = rec_ista_2d_p(
    data=kspace_data,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    mu=1e-9,
    smaps=Smaps,
    nb_scales=4,
    lambda_init=1.0,
    max_iter=max_iter,
    tol=1e-4,
    cartesian_sampling=True,
    acceleration=True,
    cost=None,
    verbose=0)
linear_op = Wavelet2(wavelet_name="BsplineWaveletTransformATrousAlgorithm",
                     nb_scale=4)

fourier_op = FFT2(samples=kspace_loc, shape=(512, 512))

image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()

# #############################################################################
# # Condata-Vu optimization
# # -----------------------
# #
# # We now want to refine the zero order solution using a Condata-Vu
# # optimization.
# # Here no cost function is set, and the optimization will reach the
# # maximum number of iterations. Fill free to play with this parameter.
#
# # Start the CONDAT-VU reconstruction
max_iter = 10

x_final, alpha_final = rec_condat_vu_2d_p(
    data=kspace_data,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    mu=1e-9,
    s_maps=Smaps,
    nb_scale=4,
    max_iter=max_iter,
    tol=1e-4,
    cartesian_sampling=True,
    cost=None,
    verbose=0)


image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
