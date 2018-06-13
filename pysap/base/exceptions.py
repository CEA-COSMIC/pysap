# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

_Exception = Exception


class Exception(_Exception):
    """ Base class for all exceptions in pysap.
    """
    def __init__(self, *args, **kwargs):
        _Exception.__init__(self, *args, **kwargs)


class Sparse2dError(Exception):
    """ Base exception type for the package.
    """
    def __init__(self, message):
        super(Sparse2dError, self).__init__(message)


class Sparse2dRuntimeError(Sparse2dError):
    """ Error thrown when call to the Sparse2d software failed.
    """
    def __init__(self, algorithm_name, parameters, error=None):
        message = (
            "Sparse2d call for '{0}' failed, with parameters: '{1}'. Error:: "
            "{2}.".format(algorithm_name, parameters, error))
        super(Sparse2dRuntimeError, self).__init__(message)


class Sparse2dConfigurationError(Sparse2dError):
    """ Error thrown when call to the Sparse2d software failed.
    """
    def __init__(self, command_name):
        message = "Sparse2d command '{0}' not found.".format(command_name)
        super(Sparse2dConfigurationError, self).__init__(message)
