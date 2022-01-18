
|CI|_ |Codecov|_ |Python36|_ |Python37|_ |Python38|_ |Python39|_ |PyPi|_ |Doc|_ |Docker|_

.. |CI| image:: https://github.com/CEA-COSMIC/pysap/workflows/CI/badge.svg
.. _CI: https://github.com/CEA-COSMIC/pysap/actions?query=workflow%3ACI

.. |Codecov| image:: https://codecov.io/gh/CEA-COSMIC/pysap/branch/master/graph/badge.svg?token=XHJIQXV7AX
.. _Codecov: https://codecov.io/gh/CEA-COSMIC/pysap

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/python-pySAP

.. |Python37| image:: https://img.shields.io/badge/python-3.7-blue.svg
.. _Python37: https://badge.fury.io/py/python-pySAP

.. |Python38| image:: https://img.shields.io/badge/python-3.8-blue.svg
.. _Python38: https://badge.fury.io/py/python-pySAP

.. |Python39| image:: https://img.shields.io/badge/python-3.9-blue.svg
.. _Python39: https://badge.fury.io/py/python-pySAP

.. |PyPi| image:: https://badge.fury.io/py/python-pySAP.svg
.. _PyPi: https://badge.fury.io/py/python-pySAP

.. |Doc| image:: https://readthedocs.org/projects/python-pysap/badge/?version=latest
.. _Doc: https://python-pysap.readthedocs.io/en/latest/?badge=latest

.. |Docker| image:: https://img.shields.io/docker/cloud/build/ceacosmic/pysap
.. _Docker: https://hub.docker.com/r/ceacosmic/pysap

PySAP
======

PySAP (Python Sparse data Analysis Package) is a Python module for **sparse data analysis** that offers:

* A common API for astronomical and neuroimaging datasets.
* Access to |link-to-sparse2d| executables with both wrappers and bindings.
* Access to application specific plug-ins.
* A graphical user interface to play with the provided functions.

.. |link-to-sparse2d| raw:: html

  <a href="https://github.com/CosmoStat/Sparse2D"
  target="_blank">Sparse2D</a>

.. raw:: html

  <img src="./doc/images/schema.jpg" width="250px">

This package is the result of the |link-to-cosmic| project, which is a collaboration between
the CEA Neurospin UNATI and CEA CosmoStat laboratories.

.. |link-to-cosmic| raw:: html

  <a href="http://cosmic.cosmostat.org/"
  target="_blank">COSMIC</a>

Important links
===============

- Official source code repo: https://github.com/cea-cosmic/pysap
- API documentation (last stable release): https://python-pysap.readthedocs.io/
- PySAP paper: https://arxiv.org/abs/1910.08465

Dependencies
============

PySAP will automatically install all of the required dependencies, however
issues may arise on some operating systems. If you encounter any problems please
ensure that you have all of the following dependencies installed before opening a
new issue.

1. PySAP requires that the COSMIC package |link-to-modopt| be installed.

.. |link-to-modopt| raw:: html

  <a href="https://github.com/CEA-COSMIC/ModOpt"
  target="_blank">ModOpt</a>

2. PySAP also requires the installation of the following third party software packages:

* scipy [>=1.5.4]
* numpy [>=1.19.5]
* matplotlib [>=3.3.4]
* astropy [>=4.1]
* nibabel [>=3.2.1]
* pyqtgraph [>=0.11.1]
* progressbar2 [>=3.53.1]
* scikit-learn [>=0.24.1]
* pybind11 [==2.6.2]
* pyqt5 [==5.15.4]
* PyWavelets [>=1.1.1]

Plug-Ins
========

PySAP currently supports the following plug-ins:

* |link-to-pysap-astro| [==0.0.1]
* |link-to-pysap-etomo| [==0.0.1]
* |link-to-pysap-mri| [==0.4.0]

.. |link-to-pysap-astro| raw:: html

  <a href="https://github.com/CEA-COSMIC/pysap-astro"
  target="_blank">PySAP-Astro</a>

.. |link-to-pysap-astro| raw:: html

  <a href="https://github.com/CEA-COSMIC/pysap-etomo"
  target="_blank">PySAP-ETomo</a>

.. |link-to-pysap-mri| raw:: html

  <a href="https://github.com/CEA-COSMIC/pysap-mri"
  target="_blank">PySAP-MRI</a>

Installation
============

The installation of PySAP has been extensively tested on Ubuntu and macOS, however
we cannot guarantee it will work on every operating system. A Docker
image is available (see below) for those unable to install PySAP directly.

If you encounter any installation issues be sure to go through the following steps before opening a new issue:

1. Check that that all the dependencies listed above have been correctly installed.
2. Read through all of the documentation provided, including the troubleshooting suggestions.
3. Check if your problem has already been addressed in a previous issue.

Further instructions are available |link-to-install|.

.. |link-to-install| raw:: html

  <a href="https://python-pysap.readthedocs.io/en/latest/generated/installation.html"
  target="_blank">here</a>

From PyPi
---------

To install PySAP simply run:

.. code-block:: bash

  $ pip install python-pysap

Depending on your Python setup you may need to provide the ``--user`` option.

.. code-block:: bash

  $ pip install --user python-pysap

Locally
-------

To build PySAP locally, clone the repository:

.. code-block:: bash

  $ git clone https://github.com/CEA-COSMIC/pysap.git

and run:

.. code-block:: bash

  $ pip install .

or:

.. code-block:: bash

  $ python setup.py install

As before, use the ``--user`` option if needed.

Custom Installation
-------------------

The following options can be passed when running ``python setup.py install``:

* ``--noplugins`` : Install PySAP without any plug-ins
* ``--only=<PLUG-IN NAME>`` : Install PySAP with only the specified plug-in names (comma separated)
* ``--nosparse2d`` : Install PySAP without building Sparse2D

For example, to install PySAP with only the Etomo plug-in and without Sparse2D
you would run the following.

.. code-block:: bash

  $ python setup.py install --nosparse2d --only=pysap-etomo

Note that these options can also be invoked when installing with ``pip`` using
the ``--install-option="<OPTION>"`` option.

.. code-block:: bash

  $ pip install . --install-option="--noplugins"

However, this will disable the use of wheels and make take significantly longer
to build all of the dependencies. Therefore, when installing PySAP this way it
is recommended to pre-install all the required dependencies or use the Conda
environment provided.

Conda Environment
-----------------

A a conda ``environment.yml`` file is provided to facilitate the installation of
the required PySAP dependencies along with some optional dependencies that
provide additional features. To build the environment run:

.. code-block:: bash

  $ conda env create -f environment.yml

Then to activate the environment run:

.. code-block:: bash

  $ conda activate pysap

Finally, install PySAP following the instructions above.

Docker Image
------------

A PySAP Docker image is available via DockerHub that includes the latest stable
version of PySAP pre-installed. To install the image run:

.. code-block:: bash

  $ docker pull ceacosmic/pysap

A Jupyter notebook can be launched using the Docker image as a backend and with
access to the users current working directory as follows:

.. code-block:: bash

  $ docker run -p 8888:8888 -v ${PWD}:/home ceacosmic/pysap

macOS
-----

Help with installation on macOS is available `here`_.

.. _here: ./doc/macos_install.rst

Linux
-----

Please refer to the |link-to-pyqt| for issues regarding the installation of
``pyqtgraph``.

.. |link-to-pyqt| raw:: html

  <a href="http://www.pyqtgraph.org/"
  target="_blank">PyQtGraph homepage</a>

Windows
-------

Help with installation on Windows on |link-to-windows-help|.

.. |link-to-windows-help| raw:: html

  <a href="https://gist.github.com/chaithyagr/4104df91fbebf44fce1589e96baa6eda"
  target="_blank">this Gist</a>

Contributing
============

If you want to contribute to pySAP, be sure to review the `contribution guidelines`_ and follow to the `code of conduct`_.

.. _contribution guidelines: ./CONTRIBUTING.md

.. _code of conduct: ./CODE_OF_CONDUCT.md


Citation
========

If you use PySAP in a scientific publication, we would appreciate citations to the following paper:
|link-to-paper|

.. |link-to-paper| raw:: html

  <a href="https://www.sciencedirect.com/science/article/pii/S2213133720300561 "target="_blank">
  PySAP: Python Sparse Data Analysis Package for multidisciplinary image processing, S. Farrens et al., Astronomy and Computing 32, 2020 </a>

The bibtex citation is the following:
k::
  @Article{farrens2020pysap,
    title={{PySAP: Python Sparse Data Analysis Package for multidisciplinary image processing}},
    author={Farrens, S and Grigis, A and El Gueddari, L and Ramzi, Z and Chaithya, GR and Starck, S and Sarthou, B and Cherkaoui, H and Ciuciu, P and Starck, J-L},
    journal={Astronomy and Computing},
    volume={32},
    pages={100402},
    year={2020},
    publisher={Elsevier}
  }
