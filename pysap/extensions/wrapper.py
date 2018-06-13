# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import os
import json
import warnings
import subprocess


# Package import
from pysap.base.exceptions import Sparse2dRuntimeError
from pysap.base.exceptions import Sparse2dConfigurationError


class Sparse2dWrapper(object):
    """ Parent class for the wrapping of Sparse2d commands.
    """
    def __init__(self, env=None, verbose=False):
        """ Initialize the Sparse2dWrapper class by setting properly the
        environment.

        Parameters
        ----------
        env: dict (optional, default None)
            the current environment in which the Sparse2d command will be
            executed. Default None, the current environment.
        verbose: bool, default False
            control the verbosity level.
        """
        self.environment = env
        self.verbose = verbose
        if env is None:
            self.environment = os.environ

    def __call__(self, cmd):
        """ Run the Sparse2d command.

        Parameters
        ----------
        cmd: list of str (mandatory)
            The command to execute.
        """
        # Check Sparse2d has been configured so the command can be found
        process = subprocess.Popen(["which", cmd[0]],
                                   env=self.environment,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.stdout, self.stderr = process.communicate()
        self.stdout = self.stdout.decode("utf-8")
        self.stderr = self.stderr.decode("utf-8")
        self.exitcode = process.returncode
        if self.exitcode != 0:
            raise Sparse2dConfigurationError(cmd[0])

        # Command must contain only strings
        _cmd = [str(elem) for elem in cmd]
        if self.verbose:
            print("[info] Executing ISAP command: {0}...".format(
                " ".join(_cmd)))

        # Execute the command
        process = subprocess.Popen(_cmd,
                                   env=self.environment,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.stdout, self.stderr = process.communicate()
        self.stdout = self.stdout.decode("utf-8")
        self.stderr = self.stderr.decode("utf-8")
        self.exitcode = process.returncode
        if self.exitcode != 0 or self.stderr or "Error" in self.stdout:
            raise Sparse2dRuntimeError(
                _cmd[0], " ".join(_cmd[1:]), self.stderr + self.stdout)
