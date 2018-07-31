"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L. El Gueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 3D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
from pysap.data import get_sample_data
from pysap.numerics.proximity import Threshold
from pysap.numerics.gradient import Gradient_pMRI
from pysap.numerics.reconstruct import sparse_rec_fista
from pysap.numerics.reconstruct import sparse_rec_condatvu
from pysap.plugins.mri.reconstruct_3D.fourier import NFFT3
from pysap.plugins.mri.parallel_mri.cost import GenericCost
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.reconstruct.utils import normalize_frequency_locations


# Third party import
import numpy as np
import matplotlib.pyplot as plt

# Load input data
Il = get_sample_data("3d-pmri")
Iref = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))

imshow3D(Iref, display=True)

samples = get_sample_data("mri-radial-3d-samples").data
samples = normalize_frequency_locations(samples)

#############################################################################
# Generate the kspace
# -------------------
#
# From the 3D phantom and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
fourier_op_gen = NFFT3(samples=samples, shape=Iref.shape)
kspace_data = fourier_op_gen.op(Iref)

# Zero order solution
image_rec0 = fourier_op_gen.adj_op(kspace_data)
imshow3D(np.abs(image_rec0), display=True)

max_iter = 5

linear_op = pyWavelet3(wavelet_name="sym4",
                       nb_scale=4)

fourier_op = NFFT3(samples=samples, shape=Iref.shape)

print('Starting Lipschitz constant computation')

gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op)

prox_op = Threshold(None)

cost_synthesis = GenericCost(
    gradient_op=gradient_op,
    prox_op=prox_op,
    linear_op=None,
    initial_cost=1e6,
    tolerance=1e-4,
    cost_interval=1,
    test_range=4,
    verbose=True,
    plot_output=None)

print('Lipschitz constant found: ', str(gradient_op.spec_rad))

x_final, transform, cost, metrics = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=cost_synthesis,
    mu=0,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)

imshow3D(np.abs(x_final), display=True)
plt.figure()
plt.plot(cost)
plt.grid(True)
plt.xlabel('Iteration number')
plt.ylabel('Cost function value')
plt.show()

gradient_op_cd = Gradient_pMRI(data=kspace_data,
                               fourier_op=fourier_op)

cost_analysis = GenericCost(
    gradient_op=gradient_op_cd,
    prox_op=prox_op,
    linear_op=linear_op,
    initial_cost=1e6,
    tolerance=1e-4,
    cost_interval=1,
    test_range=4,
    verbose=True,
    plot_output=None)

x_final, transform, cost, metrics = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    prox_dual_op=prox_op,
    cost_op=cost_analysis,
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
