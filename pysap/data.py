# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module that privides the utility functions to download toy datasets.
"""

# System import
from __future__ import print_function
import os
import sys
if sys.version_info[:2] >= (3, 0):
    from urllib.request import FancyURLopener
    from urllib.request import urlopen
    from urllib.request import urlparse
    from urllib.request import HTTPError
else:
    from urllib import FancyURLopener
    from urllib2 import urlopen
    from urllib2 import urlparse
    urlparse = urlparse.urlparse
    from urllib2 import HTTPError
import urllib
import time
import shutil
import numpy

import hashlib

# Package import
import pysap
from pysap.base.exceptions import Exception


# Global parameters
SAMPLE_DATE_FILES = {
    "mri-nifti": {
        "url": ("ftp://ftp.cea.fr/pub/unati/nsap/pysap/datasets/"
                "t1_localizer.nii.gz"),
        "md5sum": "9617b36e5510a4783038c63241da21d4"
     },
    "mri-slice-nifti": {
        "url": ("ftp://ftp.cea.fr/pub/unati/nsap/pysap/datasets/"
                "BrainPhantom512.nii.gz"),
        "md5sum": "19983e6003ae94487d03131f4bacae2e"
    },
    "mri-mask": {
        "url": ("ftp://ftp.cea.fr/pub/unati/nsap/pysap/datasets/"
                "mask_BrainPhantom512.nii.gz"),
        "md5sum": "078760d89e737e69b5578d47e368c42f"
    },
    "astro-fits": {
        "url": "ftp://ftp.cea.fr/pub/unati/nsap/pysap/datasets/M31_128.fits",
        "md5sum": "1371f06a3b7fe5588ec4823dd9f2ccad"
    },
    "astro-mask": {
        "url": ("ftp://ftp.cea.fr/pub/unati/nsap/pysap/datasets/"
                "mask25_sig40.fits"),
        "md5sum": "8d7fd9b4d7c2aaf407fa1331860a130f"
    },
    "astro-galaxy": {
        "url": ("ftp://ftp.cea.fr/pub/unati/nsap/pysap/datasets/"
                "example_galaxy_image.npy"),
        "md5sum": None
    },
    "astro-psf": {
        "url": ("ftp://ftp.cea.fr/pub/unati/nsap/pysap/datasets/"
                "example_psf_image.npy"),
        "md5sum": None
    }
}
DATADIR = os.path.join(os.path.expanduser("~"), ".local", "share", "pysap")
PACKAGEDIR = os.path.dirname(pysap.__file__)


def get_sample_data(dataset_name, datadir=DATADIR, verbose=1):
    """ Get a sample dataset.

    This function download the requested dataset in the
    '$HOME/.local/share/pysap' directory.

    Parameters
    ----------
    dataset_name: str
        which sample data you want, must be defined in the 'SAMPLE_DATE_FILES'
        dictionary.
    verbose: int (optional, default 1)
        control the verbosity level.

    Returns
    -------
    image: Image
        the loaded dataset.
    """
    # First get the data url
    dataset = SAMPLE_DATE_FILES.get(dataset_name)
    if dataset is None:
        raise Exception("No '{0}' sample data available - allowed sample data "
                        "are {1}.".format(dataset_name,
                                          SAMPLE_DATE_FILES.keys()))

    # Get the resource on the web or on the local machine
    dataset["url"] = dataset["url"].format(**{"PYSAP": PACKAGEDIR})
    if os.path.isfile(dataset["url"]):
        path = copy_file(dataset["url"], data_dir=DATADIR, overwrite=False,
                         verbose=verbose)
    else:
        path = download_file(dataset["url"], data_dir=DATADIR, resume=True,
                             overwrite=False, verbose=verbose)

    # md5 check sum
    if dataset["md5sum"] is not None:
        if (md5_sum_file(path) != dataset["md5sum"]):
            raise Exception("File '{0}' checksum verification has "
                            "failed.".format(path))

    # Load the dataset
    image = pysap.io.load(path)

    return image


def md5_sum_file(fname):
    """ Calculates the MD5 sum of a file.

    Parameters
    ----------
    fname: str (mandatory)
        the path to a file

    Returns
    -------
    md5: int
        the md5 sum of the input file
    """
    f = open(fname, 'rb')
    m = hashlib.md5()
    while True:
        data = f.read(8192)
        if not data:
            break
        m.update(data)
    return m.hexdigest()


def progress_bar(ratio, title, bar_length=20, maxsize=40):
    """ Generate a progress bar

    Parameters
    ----------
    ratio: float (mandatory)
        the progress status (0<=ratio<1)
    fname: str (optional)
        the name of the file beeing dowloaded
    bar_length: int (optional)
        the size of the progress bar
    maxsize: int (optional)
        use to justify title.
    """
    progress = int(ratio * 100.)
    block = int(round(bar_length * ratio))
    title = title.ljust(maxsize, " ")
    text = "\r[{0}] {1}% {2}".format(
        "=" * block + " " * (bar_length - block), progress, title)
    sys.stdout.write(text)
    sys.stdout.flush()


class ResumeURLOpener(FancyURLopener):
    """Create sub-class in order to overide error 206. This error means a
    partial file is being sent, which is fine in this case.
    Do nothing with this error.

    Note
    ----
    This was adapted from:
    http://code.activestate.com/recipes/83208-resuming-download-of-a-file/
    """
    def __init__(self):
        super(FancyURLopener, self).__init__()

    def http_error_206(self, url, fp, errcode, errmsg, headers, data=None):
        pass


def download_file(url, data_dir, resume=True, overwrite=False, verbose=0):
    """ Load requested file if needed or requested.

    Parameters
    ----------
    url: str
        the url of the file to be downloaded.
    data_dir: str
        path of the data directory.
    resume: bool (optional, default True)
        if True, try to resume partially downloaded files
    overwrite: bool (optional, default False)
        if True and file already exists, delete it.
    verbose: int (optional, default 0)
        control the verbosity level.

    Returns
    -------
    download_fname: str
        absolute path to the downloaded file.

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded files are
    removed.
    """
    # Create the download directory if necessary
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Determine filename using URL
    parse = urlparse(url)
    fname = os.path.basename(parse.path)

    # Generate the download file name
    download_fname = os.path.join(data_dir, fname)

    # Generate a temporary file for the download
    temp_fname = os.path.join(data_dir, fname + ".part")

    # If the file is already created remove it if the overwrite option is set
    # or return the file
    if os.path.exists(download_fname):
        if overwrite:
            os.remove(download_fname)
        else:
            return download_fname

    # If the temporary file is already created remove it if the overwrite
    # option is set
    if os.path.exists(temp_fname):
        if overwrite:
            os.remove(temp_fname)

    # Start a timer to evaluate the download time
    t0 = time.time()

    # Start downloading dataset
    local_file = None
    bytes_so_far = 0
    try:
        # Prepare the download
        if verbose > 0:
            print("Downloading data from {0}...".format(url))
        # Case 1: continue the downloading from an existing temporary file
        if resume and os.path.exists(temp_fname):
            url_opener = ResumeURLOpener()
            # Download has been interrupted, we try to resume it.
            local_file_size = os.path.getsize(temp_fname)
            # If the file exists, then only download the remainder
            url_opener.addheader("Range", "bytes={0}-".format(local_file_size))
            try:
                data = url_opener.open(url)
            except HTTPError:
                # There is a problem that may be due to resuming
                # Restart the downloading from scratch
                return download_file(url, data_dir, resume=False,
                                     overwrite=False)
            local_file = open(temp_fname, "ab")
            bytes_so_far = local_file_size
        # Case 2: just download the file
        else:
            data = urlopen(url)
            local_file = open(temp_fname, "wb")
        # Get the total file size
        try:
            if sys.version_info[:2] >= (3, 0):
                total_size = data.info().get_all("Content-Length")[0].strip()
            else:
                total_size = data.info().getheader("Content-Length").strip()
            total_size = int(total_size) + bytes_so_far
        except Exception as e:
            if verbose > 0:
                print("Total size could not be determined.")
            total_size = "?"

        # Download data
        chunk_size = 8192
        while True:
            # Read chunk
            chunk = data.read(chunk_size)
            # Stoping criterion
            if not chunk:
                break
            # Write to local file
            bytes_so_far += len(chunk)
            local_file.write(chunk)
            # Write report status and print a progress bar
            if isinstance(total_size, int):
                ratio = float(bytes_so_far) / float(total_size)
            else:
                ratio = 0
            progress_bar(ratio, title=os.path.basename(url))
        print()

        # Temporary file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        shutil.move(temp_fname, download_fname)

        # Get process duration and print it
        dt = time.time() - t0
        exit_message = ("Download was done in {0} minutes, {1: .2f} "
                        "seconds").format(int(numpy.floor(dt / 60)), dt % 60)
        if verbose > 0:
            print(exit_message)
    except HTTPError as e:
        raise Exception("{0}\nError while downloading file '{1}'. "
                        "Dataset download aborted.".format(e, fname))
    finally:
        # Temporary file must be closed
        if local_file is not None:
            if not local_file.closed:
                local_file.close()

    return download_fname


def copy_file(path, data_dir, overwrite=False, verbose=0):
    """ Copy the requested file if needed or requested.

    Parameters
    ----------
    path: str
        the path to the file to be downloaded.
    data_dir: str
        path of the data directory.
    overwrite: bool (optional, default False)
        if True and file already exists, delete it.
    verbose: int (optional, default 0)
        control the verbosity level.

    Returns
    -------
    copy_file: str
        absolute path to the copied file.
    """
    # Create the download directory if necessary
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate the download file name
    copy_fname = os.path.join(data_dir, os.path.basename(path))

    # If the file is already created remove it if the overwrite option is set
    # or return the file
    if os.path.exists(copy_fname):
        if overwrite:
            os.remove(copy_fname)
        else:
            return copy_fname

    # Start a timer to evaluate the copy time
    t0 = time.time()

    # Start copying dataset
    print("Copying data from {0}...".format(path))
    shutil.copy2(path, copy_fname)

    # Get process duration and print it
    dt = time.time() - t0
    exit_message = ("Copy was done in {0} minutes, {1: .2f} "
                    "seconds").format(int(numpy.floor(dt / 60)), dt % 60)
    if verbose > 0:
        print(exit_message)

    return copy_fname
