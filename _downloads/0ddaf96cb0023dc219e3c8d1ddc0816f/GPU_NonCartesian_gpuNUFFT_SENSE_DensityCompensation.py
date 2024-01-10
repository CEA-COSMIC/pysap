"""
GPU non-cartesian reconstruction with SENSE
===========================================

Author: Chaithya G R

In this tutorial we will reconstruct an MR Image directly with density
compensation and SENSE from gpuNUFFT

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 3D orange data
and the radial acquisition scheme (non-cartesian).
"""

# %%
# Package import
from mri.operators import NonCartesianFFT, WaveletN
from mri.operators.utils import normalize_frequency_locations
from mri.operators.fourier.utils import estimate_density_compensation
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np
import matplotlib.pyplot as plt

# %%
# Loading input data
image = get_sample_data('3d-pmri').data.astype(np.complex64)
cartesian = np.linalg.norm(image, axis=0)

# Obtain MRI non-cartesian mask and estimate the density compensation
radial_mask = get_sample_data("mri-radial-3d-samples")
kspace_loc = normalize_frequency_locations(radial_mask.data)
density_comp = estimate_density_compensation(kspace_loc, cartesian.shape, 'pipe', backend='gpunufft')

# %%
# View Input
plt.subplot(1, 2, 1)
plt.imshow(cartesian[..., 80], cmap='gray')
plt.title("MRI Data")
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(*kspace_loc[::500].T, s=0.1, alpha=0.5)
plt.title("K-space Sampling Mask")
plt.show()

# %%
# Generate the kspace
# -------------------
#
# From the 3D orange slice and 3D radial acquisition mask, we retrospectively
# undersample the k-space
# We then reconstruct using adjoint with and without density compensation

# Get the locations of the kspace samples and the associated observations
fourier_op = NonCartesianFFT(
    samples=kspace_loc,
    shape=cartesian.shape,
    n_coils=image.shape[0],
    implementation='gpuNUFFT',
)
kspace_obs = fourier_op.op(image)
# %%
# Obtrain the Sensitivity Maps
Smaps, SOS = get_Smaps(
    k_space=kspace_obs,
    img_shape=fourier_op.shape,
    samples=kspace_loc,
    thresh=(0.05, 0.05, 0.05),  # The cutoff threshold in each kspace
                                # direction between 0 and kspace_max (0.5)
    min_samples=kspace_loc.min(axis=0),
    max_samples=kspace_loc.max(axis=0),
    density_comp=density_comp,
    mode='NFFT',
)
# %%
# View Input
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.abs(Smaps[i][..., 80]), cmap='gray')
plt.suptitle("Sensitivty Maps")
plt.show()

# %%
# Density Compensation adjoint:
fourier_op_sense_dc = NonCartesianFFT(
    samples=kspace_loc,
    shape=cartesian.shape,
    implementation='gpuNUFFT',
    n_coils=image.shape[0],
    density_comp=density_comp,
    smaps=Smaps,
)
# This preconditions k-space giving a result closer to inverse
image_rec = fourier_op_sense_dc.adj_op(kspace_obs)
recon_ssim = ssim(image_rec, cartesian, mask=np.abs(image)>np.mean(np.abs(image)))
plt.imshow(np.abs(image_rec)[..., 80], cmap='gray')
plt.title('Density Compensated Adjoint : SSIM = ' + str(np.around(recon_ssim, 2)))
plt.show()


# %%
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Setup the operators
linear_op = WaveletN(
    wavelet_name='sym8',
    nb_scale=4,
    dim=3,
)
regularizer_op = SparseThreshold(Identity(), 1e-11, thresh_type="soft")
# Setup Reconstructor
reconstructor = SelfCalibrationReconstructor(
    fourier_op=fourier_op_sense_dc,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)
# %%
# Run the FISTA reconstruction and view results
image_rec, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_obs,
    optimization_alg='fista',
    num_iterations=30,
)
recon_ssim = ssim(image_rec, cartesian)
plt.imshow(np.abs(image_rec)[..., 80], cmap='gray')
plt.title('Iterative Reconstruction : SSIM = ' + str(np.around(recon_ssim, 2)))
plt.show()
