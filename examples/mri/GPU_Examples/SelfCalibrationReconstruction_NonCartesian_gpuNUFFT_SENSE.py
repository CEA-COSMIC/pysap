"""
Non Cartesian Self Calibrating Reconstruction
=============================================

Author: Chaithya G R

In this tutorial we will reconstruct an MRI image from non cartesian kspace
measurements. We will use gpuNUFFT and SENSE Reconstruction on GPU

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D parallel MRI
brain slice on 32 channels and the acquisition using non cartesian radial
samples.
"""

# Package import
from mri.operators import NonCartesianFFT, WaveletN
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
from mri.operators.utils import convert_locations_to_mask, \
    gridded_inverse_fourier_transform_nd
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np

# Loading input data
cartesian_ref_image = get_sample_data('2d-pmri')
image = pysap.Image(data=np.sqrt(np.sum(cartesian_ref_image.data**2, axis=0)))

# Obtain MRI cartesian mask
mask = get_sample_data("mri-radial-samples")
kspace_loc = mask.data

# View Input
# image.show()
# mask.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquisition mask, we retrospectively
# undersample the k-space using a non cartesian acquisition mask
# We then grid the kspace to get the gridded solution as a baseline

# Get the kspace observation values for the kspace locations
fourier_op = NonCartesianFFT(
    samples=kspace_loc,
    shape=image.shape,
    n_coils=cartesian_ref_image.shape[0],
    implementation='gpuNUFFT'
)
kspace_obs = fourier_op.op(cartesian_ref_image)

# Gridded solution
grid_space = np.linspace(-0.5, 0.5, num=image.shape[0])
grid2D = np.meshgrid(grid_space, grid_space)
grid_soln = np.asarray([
    gridded_inverse_fourier_transform_nd(kspace_loc, kspace_obs_ch,
                                         tuple(grid2D), 'linear')
    for kspace_obs_ch in kspace_obs
])
image_rec0 = pysap.Image(data=np.sqrt(np.sum(np.abs(grid_soln)**2, axis=0)))
# image_rec0.show()
base_ssim = ssim(image_rec0, image)
print('The Base SSIM is : ' + str(base_ssim))

# Obtain SMaps
Smaps, SOS = get_Smaps(
    k_space=kspace_obs,
    img_shape=fourier_op.shape,
    samples=kspace_loc,
    thresh=(0.01, 0.01),    # The cutoff threshold in each kspace direction
                            # between 0 and kspace_max (0.5)
    min_samples=kspace_loc.min(axis=0),
    max_samples=kspace_loc.max(axis=0),
    mode='gridding',
    method='linear',
    n_cpu=-1,
)
# Setup Fourier Operator with SENSE. This would initialize the
# fourier operators in the GPU.
# For this we need to specify the implementation as gpuNUFFT
# and also pass the Smaps calculated above
fourier_op_sense = NonCartesianFFT(
    samples=kspace_loc,
    shape=image.shape,
    n_coils=cartesian_ref_image.shape[0],
    smaps=Smaps,
    implementation='gpuNUFFT',
)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Setup the operators
linear_op = WaveletN(
    wavelet_name='sym8',
    nb_scale=4,
)
regularizer_op = SparseThreshold(Identity(), 3 * 1e-9, thresh_type="soft")
# Setup Reconstructor
reconstructor = SelfCalibrationReconstructor(
    fourier_op=fourier_op_sense,
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
image_rec = pysap.Image(data=x_final)
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
