
|Travis|_ |Coveralls|_ |Python36|_ |Python37|_ |Python38|_ |PyPi|_ |Doc|_ |CircleCI|_

.. |Travis| image:: https://travis-ci.org/CEA-COSMIC/pysap.svg?branch=master
.. _Travis: https://travis-ci.org/CEA-COSMIC/pysap

.. |Coveralls| image:: https://coveralls.io/repos/CEA-COSMIC/pysap/badge.svg?branch=master&service=github
.. _Coveralls: https://coveralls.io/github/CEA-COSMIC/pysap

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/python-pySAP

.. |Python37| image:: https://img.shields.io/badge/python-3.7-blue.svg
.. _Python37: https://badge.fury.io/py/python-pySAP

.. |Python38| image:: https://img.shields.io/badge/python-3.8-blue.svg
.. _Python38: https://badge.fury.io/py/python-pySAP

.. |PyPi| image:: https://badge.fury.io/py/python-pySAP.svg
.. _PyPi: https://badge.fury.io/py/python-pySAP

.. |Doc| image:: https://readthedocs.org/projects/python-pysap/badge/?version=latest
.. _Doc: https://python-pysap.readthedocs.io/en/latest/?badge=latest

.. |CircleCI| image:: https://circleci.com/gh/CEA-COSMIC/pysap.svg?style=svg
.. _CircleCI: https://circleci.com/gh/CEA-COSMIC/pysap


PySAP
======

PySAP (Python Sparse data Analysis Package) is a Python module for **sparse data analysis** that offers:

* A common API for astronomical and neuroimaging datasets.
* Access to |link-to-sparse2d| executables with both wrappers and bindings.
* A graphical user interface to play with the provided functions.

.. |link-to-sparse2d| raw:: html

  <a href="https://github.com/CosmoStat/Sparse2D"
  target="_blank">Sparse2D</a>

This package is the result of the COSMIC project, which is a collaboration between
the CEA Neurospin UNATI and CEA CosmoStat laboratories.


Important links
===============

- Official source code repo: https://github.com/cea-cosmic/pysap
- API documentation (last stable release): https://python-pysap.readthedocs.io/


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

* astropy
* matplotlib
* nibabel
* numpy
* scipy
* progressbar2
* pyqtgraph
* PyWavelets
* scikit-learn


Installation
============

The installation of PySAP has been extensively tested on Ubuntu and macOS, however
we cannot guarantee it will work on every operating system (e.g. Windows).

If you encounter any installation issues be sure to go through the following steps before opening a new issue:

1. Check that that all of the installed all the dependencies listed above have been installed.
2. Read through all of the documentation provided, including the troubleshooting suggestions.
3. Check if you problem has already been addressed in a previous issue.

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

  $ python setup.py install

or:

.. code-block:: bash

  $ python setup.py develop

As before, use the ``--user`` option if needed.

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

Contributing
============

If you want to contribute to pySAP, be sure to review the `contribution guidelines`_ and follow to the `code of conduct`_.

.. _contribution guidelines: ./CONTRIBUTING.md

.. _code of conduct: ./CODE_OF_CONDUCT.md
