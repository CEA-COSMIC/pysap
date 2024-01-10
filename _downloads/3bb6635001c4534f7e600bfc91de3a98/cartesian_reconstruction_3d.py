"""
3D Neuroimaging cartesian reconstruction
========================================

Author: LElgueddari

In this tutorial we will reconstruct an MRI image from the sparse 3D kspace
measurements.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically the 3D orange
and the cartesian acquisition scheme.
"""
# %%
# Package import
from modopt.math.metrics import ssim
from mri.operators import FFT, WaveletN
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
from pysap.data import get_sample_data
import pysap

# Third party import
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np
import matplotlib.pyplot as plt

# %%
# Loading input data and convert it into a single channel using Sum-Of-Squares
image = get_sample_data('3d-pmri').data
image = np.linalg.norm(image, axis=0)

# Obtain K-Space Cartesian Mask (straight line readout along z)
mask = get_sample_data("2d-poisson-disk-mask").data
mask = np.repeat(np.expand_dims(mask, axis=-1), image.shape[-1],
                      axis=-1)

# %%
# View Input
plt.subplot(1, 2, 1)
plt.imshow(np.abs(image[..., 80]), cmap='gray')
plt.title("MRI Data")
plt.subplot(1, 2, 2)
plt.imshow(mask[..., 80], cmap='gray')
plt.title("K-space Sampling Mask")
plt.show()

# %%
# Generate the kspace
# -------------------
#
# From the 3D Orange volume and the acquisition mask, we retrospectively
# undersample the k-space using a cartesian acquisition mask
# We then reconstruct the zero order solution as a baseline


# Generate the subsampled kspace
fourier_op = FFT(mask=mask, shape=image.shape)
kspace_data = fourier_op.op(image)

# %%
# Zero order solution
image_rec0 = fourier_op.adj_op(kspace_data)
base_ssim = ssim(image_rec0, image)
plt.imshow(np.abs(image_rec0[..., 80]), cmap='gray')
plt.title('Gridded solution : SSIM = ' + str(np.around(base_ssim, 2)))
plt.show()

# %%
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Setup the operators
linear_op = WaveletN(
    wavelet_name="sym8",
    nb_scales=4,
    dim=3,
    padding_mode="periodization",
)
regularizer_op = SparseThreshold(Identity(), 2 * 1e-11, thresh_type="soft")
# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)
# %%
# Start Reconstruction
image_rec, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_data,
    optimization_alg='fista',
    num_iterations=200,
)
recon_ssim = ssim(image_rec, image)
plt.imshow(np.abs(image_rec[..., 80]), cmap='gray')
plt.title('Iterative Reconstruction : SSIM = ' + str(np.around(recon_ssim, 2)))
plt.show()
