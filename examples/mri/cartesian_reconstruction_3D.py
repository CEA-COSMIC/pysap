"""
Neuroimaging cartesian reconstruction
=====================================

Credit: S Lannuzel, L Elgueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 3D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pysap.plugins.mri.reconstruct_3D.utils import convert_locations_to_mask
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pysap.plugins.mri.parallel_mri.gradient import Grad_pMRI
from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.reconstruct_3D.fourier import FFT3
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu


# Third party import
import numpy as np
import scipy.fftpack as pfft
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load input data
filename = '/neurospin/optimed/LoubnaElGueddari/p_MRI_CSGRE_3D/' \
            '2018-01-15_32ch_ref_nc1000_data/' \
            'meas_MID14_gre_800um_iso_128x128x128_FID24.mat'
Iref = loadmat(filename)['ref']

imshow3D(Iref, display=True)

samples = loadmat('/volatile/data/sampling_schemes/'
                  'samples_sparkling_3D_N128_502x1536x8_FID4971.mat')['samples']
samples = normalize_samples(samples)
cartesian_samples = convert_locations_to_mask(samples, [128, 128, 128])
imshow3D(cartesian_samples, display=True)

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
kspace_mask = pfft.ifftshift(cartesian_samples)
kspace_data = pfft.fftn(Iref) * kspace_mask

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations_3D(kspace_mask)

# Zero order solution
image_rec0 = pfft.ifftn(kspace_data)
imshow3D(np.abs(image_rec0), display=False)

max_iter = 10

linear_op = pyWavelet3(wavelet_name="sym4",
                       nb_scale=4)

fourier_op = FFT3(samples=kspace_loc, shape=Iref.shape)
gradient_op = Grad_pMRI(data=kspace_data,
                        fourier_op=fourier_op,
                        linear_op=linear_op)

x_final, transform, cost = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    mu=5e-2,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1,
    get_cost=True)
imshow3D(np.abs(x_final), display=True)
plt.figure()
plt.plot(cost)
plt.show()

gradient_op_cd = Grad_pMRI(data=kspace_data,
                           fourier_op=fourier_op)
x_final, transform = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    std_est=None,
    std_est_method="dual",
    std_thr=2.,
    mu=1e-5,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=0,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)

imshow3D(np.abs(x_final), display=True)
