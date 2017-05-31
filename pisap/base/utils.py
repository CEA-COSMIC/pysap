##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module define usefull helper.
"""
import os
import time
import tempfile
import numpy as np
from functools import wraps
from pisap.base.exceptions import Sparse2dRuntimeError
import pisap


def retry(exceptions=(Sparse2dRuntimeError), nb_try=10, mdelay=1.0):
    """ Decorator to run ``nb_try`` the given function.
    """
    def deco_retry(func):
        @wraps(func)
        def wrap_func(*args, **kwargs):
            i = 0
            while True:
                try:
                    res = func(*args, **kwargs)
                    break
                except exceptions:
                    i = i + 1
                    time.sleep(mdelay)
                    if i >= nb_try:
                        raise
            return res
        return wrap_func
    return deco_retry


@retry()
def stable_mr_transform(in_image, out_mr_file, **kwargs):
    """ Re-run multiple time till no exceptions occurs.
    """
    return pisap.extensions.mr_transform(in_image, out_mr_file, **kwargs)


@retry()
def stable_mr_recons(in_image, out_mr_file):
    """ Re-run multiple time till no exceptions occurs.
    """
    return pisap.extensions.mr_recons(in_image, out_mr_file)


def set_bands_shapes(bands_lengths, ratio=None):
    """ Return the bands_shapes attributs from the given bands_lengths.
    """
    if ratio is None:
        ratio = np.ones_like(bands_lengths)
    bands_shapes = []
    for ks, scale in enumerate(bands_lengths):
        scale_shapes = []
        for kb, padd in enumerate(scale):
            shape = (int(np.sqrt(padd*ratio[ks, kb])), int(np.sqrt(padd/ratio[ks, kb])))
            scale_shapes.append(shape)
        bands_shapes.append(scale_shapes)
    return bands_shapes


def to_2d_array(a):
    """Convert:
        2d array to 2d array
        1d array to 2d array
        scalar to 2d array.

        Parameters:
        ----------
        a: matplotlib axe or np.ndarray of matplotlib axe (1d or 2d)

        Note:
        -----
        no check done in the function
    """
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            return a
        elif a.ndim == 1:
            return a[:, None]
    else:
        return np.array(a)[None, None]


def secure_mkdir(dirname):
    """ Silently pass if directory ``dirname`` already exist.
    """
    try:
        os.mkdir(dirname)
    except OSError:
        pass


def _isap_transform(data, **kwargs):
    """ Helper return the transformation coefficient in a isap-cube format.
    """
    tmpdir = tempfile.mkdtemp()
    in_image = os.path.join(tmpdir, "in.fits")
    out_mr_file = os.path.join(tmpdir, "cube.mr")
    try:
        pisap.io.save(data, in_image)
        stable_mr_transform(in_image, out_mr_file, **kwargs)
        image = pisap.io.load(out_mr_file)
        isap_trf_buf = image.data
        header = image.metadata
    except:
        raise
    finally:
        for path in (in_image, out_mr_file):
            if os.path.isfile(path):
                os.remove(path)
        os.rmdir(tmpdir)
    return isap_trf_buf, header


def isap_transform(data, **kwargs):
    """ Return the transformation coefficient in a isap-cube format.
    """
    if np.any(np.iscomplex(data)):
        isap_trf_buf_r, header = _isap_transform(data.real, **kwargs)
        isap_trf_buf_i, _ = _isap_transform(data.imag, **kwargs)
        return isap_trf_buf_r + 1.j * isap_trf_buf_i, header
    else:
        return _isap_transform(data.astype(float), **kwargs)


def _isap_recons(data, header):
    """ Helper return the reconstructed image.
    """
    cube = pisap.Image(data=data, metadata=header)
    tmpdir = tempfile.mkdtemp()
    in_mr_file = os.path.join(tmpdir, "cube.mr")
    out_image = os.path.join(tmpdir, "out.fits")
    try:
        pisap.io.save(cube, in_mr_file)
        stable_mr_recons(in_mr_file, out_image)
        isap_recs_buf = pisap.io.load(out_image)
    except:
        raise
    finally:
        for path in (in_mr_file, out_image):
            if os.path.isfile(path):
                os.remove(path)
        os.rmdir(tmpdir)
    return isap_recs_buf


def isap_recons(data, header):
    """ Return the reconstructed image.
    """
    if np.any(np.iscomplex(data)):
        isap_recs_buf_r = _isap_recons(data.real, header)
        isap_recs_buf_i = _isap_recons(data.imag, header)
        return isap_recs_buf_r + 1.j * isap_recs_buf_i
    else:
        return _isap_recons(data.astype(float), header)


def run_both(linear_op, data, nb_scale, isap_kwargs):
    """ Run ispa and pisap trf and reconstruction.
    """
    init_kwarg = {'maxscale': nb_scale}
    linear_op = linear_op(**init_kwarg)
    trf = linear_op.op(data)
    trf_img = trf.to_cube()
    trf_isap_img, header =  isap_transform(data, **isap_kwargs)
    recs_img = linear_op.adj_op(trf)
    recs_isap_img = isap_recons(trf.to_cube(), header)
    return (trf_img, trf_isap_img), (recs_img, recs_isap_img)
