"""
Neuroimaging non-cartesian reconstruction
=========================================

Author: Chaithya G R

In this tutorial we will reconstruct an MRI image from non-cartesian kspace
measurements.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquisition cartesian scheme.
"""

# Package import
from mri.operators import NonCartesianFFT, WaveletUD2
from mri.operators.utils import convert_locations_to_mask, \
    gridded_inverse_fourier_transform_nd
from mri.reconstructors import SingleChannelReconstructor
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np

# Loading input data
image = get_sample_data('2d-mri')

# Obtain MRI non-cartesian mask
radial_mask = get_sample_data("mri-radial-samples")
kspace_loc = radial_mask.data
mask = pysap.Image(data=convert_locations_to_mask(kspace_loc, image.shape))

# View Input
# image.show()
# mask.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquisition mask, we retrospectively
# undersample the k-space using a radial acquisition mask
# We then reconstruct the zero order solution as a baseline

# Get the locations of the kspace samples and the associated observations
fourier_op = NonCartesianFFT(samples=kspace_loc, shape=image.shape,
                             implementation='cpu')
kspace_obs = fourier_op.op(image.data)

# Gridded solution
grid_space = np.linspace(-0.5, 0.5, num=image.shape[0])
grid2D = np.meshgrid(grid_space, grid_space)
grid_soln = gridded_inverse_fourier_transform_nd(kspace_loc, kspace_obs,
                                                 tuple(grid2D), 'linear')
image_rec0 = pysap.Image(data=grid_soln)
# image_rec0.show()
base_ssim = ssim(image_rec0, image)
print('The Base SSIM is : ' + str(base_ssim))

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Setup the operators
linear_op = WaveletUD2(
    wavelet_id=24,
    nb_scale=4,
)
regularizer_op = SparseThreshold(Identity(), 6 * 1e-7, thresh_type="soft")
# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)
x_final, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_obs,
    optimization_alg='fista',
    num_iterations=200,
)
image_rec = pysap.Image(data=np.abs(x_final))
# image_rec.show()
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
