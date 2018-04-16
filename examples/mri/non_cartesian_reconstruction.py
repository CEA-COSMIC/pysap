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
from pysap.plugins.mri.reconstruct.fourier import NFFT2
from pysap.plugins.mri.opt import rec_ista_2d, rec_condat_vu_2d
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations

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
fourier_op = NFFT2(samples=kspace_loc, shape=image.shape)
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

# Start the FISTA reconstruction
max_iter = 20
x_final = rec_ista_2d(
    data=kspace_obs,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    mu=1e-9,
    nb_scales=4,
    max_iter=max_iter,
    tol=1e-4,
    cartesian_sampling=False,
    uniform_data_shape=image.shape,
    acceleration=True,
    cost='auto',
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
max_iter = 20
x_final, transform = rec_condat_vu_2d(
    data=kspace_obs,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    nb_scales=4,
    mu=1e-5,
    tol=1e-4,
    max_iter=max_iter,
    cartesian_sampling=False,
    uniform_data_shape=image.shape,
    cost='auto',
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
