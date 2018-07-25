# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module some functions used for the dictionary learning Compressed Sensing
reconstruction.
"""

# System import
from __future__ import division
import time
import itertools
import random

# Third party import
import numpy as np
import progressbar
from sklearn.utils import check_random_state, gen_batches
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d


def timer(start, end):
    """ Give duration time between 2 times in hh:mm:ss.
    Parameters
    ----------
    start: float
        the starting time.
    end: float
        the ending time.
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                           int(minutes),
                                           seconds))


def min_max_normalize(img):
    """ Center and normalize the given array.
    Parameters
    ----------
    img: np.ndarray
    """
    img = np.nan_to_num(img)
    min_img = img.min()
    max_img = img.max()
    img = (img - min_img) / (max_img - min_img)
    return np.nan_to_num(img)


def extract_patches_from_2d_images(img, patch_shape):
    """ Return the flattened patches from the 2d image.

    Parameters
    ----------
        img: np.ndarray of floats, the input 2d image
        patch_shape: tuple of int, shape of the patches
    Returns
    -------
        patches: np.ndarray of floats, a 2d matrix with
        -        dim nb_patches*(patch.shape[0]*patch_shape[1])
    """
    patches = extract_patches_2d(img, patch_shape)
    patches = patches.reshape(patches.shape[0], -1)
    return patches


def generate_flat_patches(images, patch_size, option='real'):
    """ Generate flat patches from the real/imaginary/complex images from the
    list of images

    Parameters
    ----------
        image: list of list of np.ndarray of float or complex
            a sublist containing all the images for one subject
        patch_size: int,
            width of square patches
        option: 'real' (default),
            'imag' real/imaginary part or 'complex'
    Return
    ------
        flat_patches: list of np.ndarray as a GENERATOR
            The patches flat and concatained as a list
    """
    patch_shape = (patch_size, patch_size)
    flat_patches = images[:]
    for imgs in flat_patches:
        flat_patches_sub = []
        for img in imgs:
            if option == "abs":
                image = np.abs(img).astype("float")
                patches = extract_patches_from_2d_images(
                    min_max_normalize(image),
                    patch_shape)
            elif option == "real":
                image = np.real(img)
                patches = extract_patches_from_2d_images(
                    min_max_normalize(image),
                    patch_shape)
            elif option == "imag":
                image = np.imag(img)
                patches = extract_patches_from_2d_images(
                    min_max_normalize(image),
                    patch_shape)
            else:
                patches_r = extract_patches_from_2d_images(
                    min_max_normalize(np.real(img)),
                    patch_shape)
                patches_i = extract_patches_from_2d_images(
                    min_max_normalize(np.imag(img)),
                    patch_shape)
                patches = patches_r + 1j * patches_i

            flat_patches_sub.append(patches)
        yield flat_patches_sub


def learn_dictionary(flat_patches_subjects, nb_atoms=100, alpha=1, n_iter=1,
                     fit_algorithm='lars', transform_algorithm='lasso_lars',
                     batch_size=100, n_jobs=1, verbose=1):
    """ Learn the dictionary from the real/imaginary part or complex paches
    from a training set

    Parameters
    ----------
    flat_patches: generator of 1d array of flat patches (floats)
            a list per subject
    nb_atoms: int,
        number of components of the dictionary (default=100)
    alpha: float,
        regulation term (default=1)
    n_iter: int
        number of iterations (default=1)
    fit_algorithm: 'lars'
        for more details see
        MiniBatchDictionaryLearning from the sklearn library
    transform_algorithm: 'lasso_lars',
        for more details see
        MiniBatchDictionaryLearning from the sklearn library
    batch_size: int (default 100),
        number of patches taken per iteration to fit the model
    n_jobs: int defaul 6,
        number of cpu to run the learning
    verbose: int default1,
        The level of verbosity

    Return
    ------
        dico: MiniBatchDictionaryLearning object
    """
    dico = MiniBatchDictionaryLearning(
        n_components=nb_atoms,
        alpha=alpha,
        n_iter=n_iter,
        fit_algorithm=fit_algorithm,
        transform_algorithm=transform_algorithm,
        n_jobs=n_jobs,
        verbose=0)
    rng = check_random_state(0)
    if verbose == 2:
        print("Dictionary Learning starting")
    t_start = time.time()
    for patches_subject in flat_patches_subjects:
        patches = list(itertools.chain(*patches_subject))
        if verbose == 1:
            print("[info] number of patches of the subject: {0}".format(
                len(patches)))
        rng.shuffle(patches)
        batches = gen_batches(len(patches), batch_size)
        nb_batches = len(patches) // batch_size
        with progressbar.ProgressBar(max_value=nb_batches,
                                     redirect_stdout=True) as bar:
            for cnt, batch in enumerate(batches):
                t0 = time.time()
                dico.partial_fit(patches[batch][:1])
                duration = time.time() - t0
                if verbose == 2:
                    print("[info] batch time: {0}".format(duration))
                bar.update(cnt)
    t_end = time.time()
    if verbose == 1:
        print("[info] dictionary learnt in {0}".format(timer(t_start, t_end)))
    return dico
