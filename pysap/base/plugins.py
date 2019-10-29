# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import sys
import re
import os
import imp


class PluginsMetaImportHook(object):
    """ A class that import a module like normal except for the plugins that
    monted on the pysap plugins module. The return value from the hack call
    is put into sys.modules.
    """
    def __init__(self):
        self.module = None

    def find_module(self, name, path=None):
        """ This method is called by Python if this class is on sys.path.
        'name' is the fully-qualified name of the module to look for, and
        'path' is either __path__ (for submodules and subpackages) or None (for
        a top-level module/package).

        Note that this method will be called every time an import statement
        is detected (or __import__ is called), before Python's built-in
        package/module-finding code kicks in.
        """
        # Use this loader only on registered modules
        match = re.match("pysap\.plugins\.(.*)", name)
        if match is None:
            return None
        name = match.groups()[0]
        if (len(name.split(".")) == 1):
            path = None

        # Get parent module and associated sub module names
        self.sub_name = name.split(".")[-1]
        self.mod_name = name.rpartition(".")[0]

        # Find the sub module and build the module path
        # TODO: use importlib.util.find_spec for Python >= 3.5 only
        try:
            self.file, self.filename, self.stuff = imp.find_module(
                self.sub_name, path)
            self.path = [self.filename]
        except ImportError:
            return None

        # Return The loader, here the object itself
        return self

    def load_module(self, name):
        """ This method is called by Python if the class
        'find_module' does not return None. 'name' is the fully-qualified name
        of the module/package that was requested.
        """
        # Load the module
        module = imp.load_module(name, self.file, self.filename, self.stuff)
        if self.file:
            self.file.close()

        # Update the module required information
        module.__path__ = self.path
        module.__loader__ = self
        module.__package__ = name
        module.__name__ = name
        if self.stuff[0] == ".py":
            module.__file__ = self.path[0]
        else:
            module.__file__ = os.path.join(self.path[0], "__init__.py")

        return module


sys.meta_path.insert(0, PluginsMetaImportHook())
