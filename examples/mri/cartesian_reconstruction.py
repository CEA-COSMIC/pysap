"""
Neuroimaging cartesian reconstruction
=====================================

Credit: A Grigis, L Elgueddari, H Carrie

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pISAP, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
import pisap
from pisap.data import get_sample_data
from pisap.plugins.mri.reconstruct.reconstruct import sparse_rec_fista
from pisap.plugins.mri.reconstruct.reconstruct import sparse_rec_condatvu
from pisap.plugins.mri.reconstruct.utils import convert_mask_to_locations

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


# Generate the subsampled kspace
kspace_mask = pfft.ifftshift(mask.data)
kspace_data = pfft.fft2(image.data) * kspace_mask

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(kspace_mask)

# Zero order solution
image_rec0 = pisap.Image(data=pfft.ifft2(kspace_data), metadata=image.metadata)
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
x_final, transform = sparse_rec_fista(
    data=kspace_data,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    mu=1e-9,
    nb_scales=4,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pisap.Image(data=np.abs(x_final))
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
x_final, transform = sparse_rec_condatvu(
    data=kspace_data,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    nb_scales=4,
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
image_rec = pisap.Image(data=np.abs(x_final))
image_rec.show()
