"""
Neuroimaging non-cartesian reconstruction
=========================================

Author: Chaithya G R

In this tutorial we will reconstruct an MR Image directly with density
compensation

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the radial acquisition scheme (non-cartesian).
"""

# Package import
from mri.operators import NonCartesianFFT, WaveletUD2
from mri.operators.utils import convert_locations_to_mask, \
    gridded_inverse_fourier_transform_nd
from mri.operators.fourier.utils import estimate_density_compensation
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

# Obtain MRI non-cartesian mask and estimate the density compensation
radial_mask = get_sample_data("mri-radial-samples")
kspace_loc = radial_mask.data
density_comp = estimate_density_compensation(kspace_loc, image.shape)

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquisition mask, we retrospectively
# undersample the k-space using a radial acquisition mask
# We then reconstruct using adjoint with and without density compensation

# Get the locations of the kspace samples and the associated observations
fourier_op = NonCartesianFFT(
    samples=kspace_loc,
    shape=image.shape,
    implementation='gpuNUFFT',
)
fourier_op_density_comp = NonCartesianFFT(
    samples=kspace_loc,
    shape=image.shape,
    implementation='gpuNUFFT',
    density_comp=density_comp
)
# Get the kspace data retrospectively. Note that this can be done with
# `fourier_op_density_comp` as the forward operator is the same
kspace_obs = fourier_op.op(image.data)

# Simple adjoint
image_rec0 = pysap.Image(data=np.abs(fourier_op.adj_op(kspace_obs)))
# image_rec0.show()
base_ssim = ssim(image_rec0, image)
print('The SSIM from Adjoint is : ' + str(base_ssim))

# Density Compensation adjoint:
# This preconditions k-space giving a result closer to inverse
image_rec1 = pysap.Image(data=np.abs(
    fourier_op_density_comp.adj_op(kspace_obs))
)
# image_rec1.show()
new_ssim = ssim(image_rec1, image)
print('The SSIM from Density '
      'compensated Adjoint is : ' + str(new_ssim))
