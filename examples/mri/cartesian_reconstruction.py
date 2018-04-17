"""
Neuroimaging cartesian reconstruction
=====================================

Credit: A Grigis, L Elgueddari, H Carrie

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
from pysap.plugins.mri.opt import rec_ista_2d, rec_condat_vu_2d
from pysap.plugins.mri.reconstruct.reconstruct import sparse_rec_condatvu
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


# Generate the subsampled kspace
kspace_mask = pfft.ifftshift(mask.data)
kspace_data = pfft.fft2(image.data) * kspace_mask

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(kspace_mask)

# Zero order solution
image_rec0 = pysap.Image(data=pfft.ifft2(kspace_data), metadata=image.metadata)
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
    data=kspace_data,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    mu=1e-7,
    nb_scales=4,
    max_iter=max_iter,
    tol=1e-4,
    cartesian_sampling=True,
    use_acceleration=False,
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

x_final, y_final = rec_condat_vu_2d(
    data=kspace_data,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    nb_scale=4,
    mu=1e-9,
    tol=1e-5,
    max_iter=max_iter,
    cost='auto',
    verbose=0)

image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
