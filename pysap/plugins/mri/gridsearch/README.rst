Repository of "EUSIPCO" paper:
##############################


Authors
-------

Hamza Cherkaoui

Philippe Ciuciu


Synopsis
--------

This repository is dedicated to reproduce the results in the
"EUSIPCO" paper.


Dependencies
------------

* alt-pisap  


Instructions
------------

Edit the configuration file:

.. code-block:: bash

    gedit config.ini

Launch the multiple reconstructions (long long time...):

.. code-block:: bash

    ./study_launcher.py -v

Produce the plots:

.. code-block:: bash

    ./post_processing.py -v output_results/ output_analysis/

