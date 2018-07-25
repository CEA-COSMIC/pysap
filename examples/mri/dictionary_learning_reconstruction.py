"""
Neuroimaging cartesian reconstruction using dictionnary Learning
================================================================

Credit: H Carrie, L Elgueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments using a dictionary learned off-line.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from pysap.numerics.fourier import FFT2
from pysap.numerics.gradient import GradAnalysis2
from pysap.numerics.reconstruct import sparse_rec_condatvu
from pysap.numerics.proximity import Threshold
from pysap.numerics.utils import convert_mask_to_locations
from pysap.numerics.cost import DualGapCost
from pysap.plugins.mri.dictionary_learning.utils import learn_dictionary
from pysap.plugins.mri.dictionary_learning.utils import min_max_normalize
from pysap.plugins.mri.dictionary_learning.linear import DictionaryLearning
from pysap.plugins.mri.dictionary_learning.utils import generate_flat_patches


# Third party import
import numpy as np
import scipy.fftpack as pfft
import matplotlib.pyplot as plt


#############################################################################
# Load the database to train the dictionary
# -----------------------------------------
#
# The database should be a list of list of 2D MRI images grouped by subjects.
# In this tutorial, the database contains only brain images.
# Besides, all images have same size and all subjects have the same
# number of slices.

dataset = get_sample_data("dict-learn-dataset")
training_set = dataset.data

print("[info]: # subjects: {0}, # images per subject: {1},"
      " image shape: ({2},{3})".format(len(training_set),
                                       len(training_set[0]),
                                       len(training_set[0][0]),
                                       len(training_set[0][0][0])))


#############################################################################
# Learning the dictionary
# -----------------------
#
# The dictionary parameters are first set: the number of components/atoms,
# the regularization term (of the Lasso) and the patch size.
# Some other parameters like the number of iterations, the batch size and
# the number of cpu running can be tuned to faster the procedure.
# Then, we will learn the complex model as 2 real-valued dictionaries
# since Scikit-learn doesn't support complex-valued data.
# We will thus learn one model for the real part and one for the imaginary
# part.

n_components = 16  # number of atoms or components
alpha = 1  # regularization term
patch_size = 7

n_iter = 1
batch_size = 10000
n_jobs = 2

# Real part
flat_patches_real = generate_flat_patches(training_set, patch_size, "real")
dico_real = learn_dictionary(flat_patches_subjects=flat_patches_real,
                             nb_atoms=n_components,
                             alpha=alpha,
                             n_iter=n_iter,
                             fit_algorithm="lars",
                             transform_algorithm="lars",
                             batch_size=batch_size,
                             n_jobs=n_jobs,
                             verbose=1)

# Imaginary part
flat_patches_imag = generate_flat_patches(training_set, patch_size, "imag")
dico_imag = learn_dictionary(flat_patches_subjects=flat_patches_imag,
                             nb_atoms=n_components,
                             alpha=alpha,
                             n_iter=n_iter,
                             fit_algorithm="lars",
                             transform_algorithm="lars",
                             batch_size=batch_size,
                             n_jobs=n_jobs,
                             verbose=1)

#############################################################################
# Plot the dictionary
# -------------------
#
# Visualize the learnt atoms.
# Let's note that you should adapt the way you dislay the atoms if you don't
# choose a perfect square number for n_components.

atoms = np.abs(dico_real.components_ + 1j * dico_imag.components_)

patch_shape = (patch_size, patch_size)
plt.figure(figsize=(4.2, 4))
for i, patch in enumerate(atoms):
    plt.subplot(np.int(np.sqrt(n_components)),
                np.int(np.sqrt(n_components)), i + 1)
    plt.imshow(patch.reshape(patch_shape), cmap=plt.cm.gray,
               interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle("Magnitude of the learnt atoms", fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()


#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Loading testing data
image = get_sample_data("mri-slice-nifti")
mask = get_sample_data("mri-mask")
image.show()
mask.show()

# Generate the subsampled kspace
kspace_mask = pfft.ifftshift(mask.data)
kspace_data = pfft.fft2(image.data) * kspace_mask

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(kspace_mask)

# Zero order solution
image_rec0 = pysap.Image(data=pfft.ifft2(kspace_data), metadata=image.metadata)
image_rec0.show()


#############################################################################
# Setting the reconstruction parameters
# -------------------------------------

fourier_op = FFT2(kspace_loc, image.data.shape)
gradient_op = GradAnalysis2(
    data=kspace_data,
    fourier_op=fourier_op)
linear_op = DictionaryLearning(
    img_shape=image.data.shape,
    dictionary_r=dico_real,
    dictionary_i=dico_imag)
prox_dual_op = Threshold(None)
cost_op = DualGapCost(
    linear_op=linear_op,
    initial_cost=1e6,
    tolerance=1e-4,
    cost_interval=1,
    test_range=4,
    verbose=0,
    plot_output=None)


#############################################################################
# Starting the reconstruction
# ---------------------------
#
# We now want to refine the zero order solution using a Condat-Vu
# optimization.
# We can't use a FISTA optimisation, which is adapted to solve the synthesis
# formulation of a Lasso.
# But here, we need to use the analysis formulation since the learnt
# dictionary is not a basis anymore. It is redundant.

x, y, costs, metrics = sparse_rec_condatvu(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_dual_op=prox_dual_op,
    cost_op=cost_op,
    std_est=None,
    std_est_method=None,
    std_thr=2.,
    mu=0.1,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=0,
    max_nb_of_iter=2,
    add_positivity=False,
    atol=1e-4,
    verbose=1)


#############################################################################
# Visualizing the results
# -----------------------

# Solution in the image domain
image_rec = pysap.Image(data=min_max_normalize(np.abs(x)))
image_rec.show()

# Solution in the sparse domain
image_sparse_rec = pysap.Image(data=min_max_normalize(
    np.abs(linear_op.adj_op(y))))
image_sparse_rec.show()
