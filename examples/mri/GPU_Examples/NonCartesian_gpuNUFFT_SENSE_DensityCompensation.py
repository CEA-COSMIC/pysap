"""
Neuroimaging non-cartesian reconstruction
=========================================

Author: Chaithya G R

In this tutorial we will reconstruct an MR Image directly with density
compensation and SENSE from gpuNUFFT

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 3D orange data
and the radial acquisition scheme (non-cartesian).
"""

# Package import
from mri.operators import NonCartesianFFT, WaveletUD2
from mri.operators.utils import convert_locations_to_mask, \
    gridded_inverse_fourier_transform_nd
from mri.operators.fourier.utils import estimate_density_compensation
from mri.reconstructors import SingleChannelReconstructor
from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np

# Loading input data
image = get_sample_data('3d-pmri')
cartesian = np.linalg.norm(image, axis=0)

# Obtain MRI non-cartesian mask and estimate the density compensation
radial_mask = get_sample_data("mri-radial-3d-samples")
kspace_loc = radial_mask.data
density_comp = estimate_density_compensation(kspace_loc, cartesian.shape)

#############################################################################
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
kspace_obs = fourier_op.op(image.data)
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
fourier_op_sense_dc = NonCartesianFFT(
    samples=kspace_loc,
    shape=cartesian.shape,
    implementation='gpuNUFFT',
    n_coils=image.shape[0],
    density_comp=density_comp,
    smaps=Smaps,
)

# Density Compensation SENSE adjoint:
# This preconditions k-space giving a result closer to inverse
image_rec1 = pysap.Image(data=np.abs(
    fourier_op_sense_dc.adj_op(kspace_obs))
)
# image_rec1.show()
base_ssim = ssim(image_rec1, cartesian)
print('The SSIM for simple Density '
      'compensated SENSE Adjoint is : ' + str(base_ssim))
