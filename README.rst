
|Travis|_ |Coveralls|_ |Python35|_ |Python36|_ |Python37|_ |PyPi|_ |Doc|_ |CircleCI|_

.. |Travis| image:: https://travis-ci.org/CEA-COSMIC/pysap.svg?branch=master
.. _Travis: https://travis-ci.org/CEA-COSMIC/pysap

.. |Coveralls| image:: https://coveralls.io/repos/CEA-COSMIC/pysap/badge.svg?branch=master&service=github
.. _Coveralls: https://coveralls.io/github/CEA-COSMIC/pysap

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/python-pySAP

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/python-pySAP

.. |Python37| image:: https://img.shields.io/badge/python-3.7-blue.svg
.. _Python37: https://badge.fury.io/py/python-pySAP

.. |PyPi| image:: https://badge.fury.io/py/python-pySAP.svg
.. _PyPi: https://badge.fury.io/py/python-pySAP

.. |Doc| image:: https://readthedocs.org/projects/python-pysap/badge/?version=latest
.. _Doc: https://python-pysap.readthedocs.io/en/latest/?badge=latest

.. |CircleCI| image:: https://circleci.com/gh/CEA-COSMIC/pysap.svg?style=svg
.. _CircleCI: https://circleci.com/gh/CEA-COSMIC/pysap



pySAP
======

Python Sparse data Analysis Package.

It levarages the `iSAP <http://www.cosmostat.org/software/isap>`_ IDL toolbox
for sparsity with applications in astrophysics or MRI.

This work is made available by a community of people, amoung which the
CEA Neurospin UNATI and CEA CosmoStat laboratories, in particular A. Grigis,
J.-L. Starck, P. Ciuciu, and S. Farrens.


Important links
===============

- Official source code repo: https://github.com/cea-cosmic/pysap
- HTML documentation (last stable release): http://cea-cosmic.github.io/pysap


Dependencies
============

The required dependencies to use the software are:

* scipy
* numpy
* matplotlib
* future
* astropy
* nibabel
* pyqtgraph
* progressbar2
* modopt

This package will generate the pysparse ISAP binding module.


Install
=======

Make sure you have installed all the dependencies listed above properly.
Further instructions are available at
https://cea-cosmic.github.io/pysap/generated/installation.html


Mac OS
------

Help with installation on Mac OS is available `here`_.

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

If you want to contribute to pySAP, be sure to review the [contribution guidelines](./CONTRIBUTING.md) and follow to the [code of conduct](.CODE_OF_CONDUCT.md).
