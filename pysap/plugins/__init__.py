# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module defines all the declared plugins.
It also propose a function to decalre external plugins.
"""

# System import
from __future__ import print_function
import os
import sys
import json
import glob
import zipfile
import importlib
import subprocess


# Global parameters
EXTPLUGINS_DIR = os.path.join(
    os.path.expanduser("~"), ".local", "share", "pysap", "extplugins")
EXTPLUGINS_FILE = os.path.join(EXTPLUGINS_DIR, "pysap-extplugins.json")
if not os.path.isfile(EXTPLUGINS_FILE):
    if not os.path.isdir(EXTPLUGINS_DIR):
        os.makedirs(EXTPLUGINS_DIR)
    with open(EXTPLUGINS_FILE, "wt") as open_file:
        open_file.write("{}")


def addplugin(location, update=False):
    """ Add a new plugin.

    Download plugins from:
    - PyPI using module names.
    - VCS project urls.
    - local directories.

    A plugin is a Python module (ie with an __init__.py file that makes it
    importable) with at least a plugin description file named
    <myplugin>.pysap-extplugin where <myplugin> is the name of the Python
    module. This description file is a JSON dictionary containing the following
    keys:
    - author: the author of the plugin.
    - version: the current version of the plugin.
    - description: a small description of the plugin.

    Parameters
    ----------
    location: str
        the PyPI registered project name or
        the location of the plugin to be added with repository type as a
        prefix. Installing from Git, Mercurial, Subversion and Bazaar and
        detects the type of VCS using url prefixes: 'git+', 'hg+', 'bzr+',
        'svn+' or
        the directory of the plugin.
    update: bool, optional
        if specified force the update of the plugin.

    Returns
    -------
    status: int
        the command status: 0 imported, 1 already imported.
    """
    # Check that the pysap plugins directory is in write mode
    pysap_plugin_dir = os.path.dirname(__file__)
    if not os.access(pysap_plugin_dir, os.W_OK):
        raise ValueError(
            "For the moment the external plugin mechanism only works with "
            "user installation. Please use the 'pip install --user' directive "
            "to install pysap.")

    # Add a plugin from a pip source
    status = 0
    if not os.path.isdir(location):
        cmd = ["pip", "download", "--no-cache-dir", "--no-deps", "-d",
               EXTPLUGINS_DIR, location]
        subprocess.check_call(cmd)
        plugins = glob.glob(os.path.join(EXTPLUGINS_DIR, "*.zip"))
        if len(plugins) != 1:
            raise ValueError("Expect exactly one pending module in "
                             "'{0}'.".format(EXTPLUGINS_DIR))
        plugin = plugins[0]
        plugin_basename = os.path.basename(plugin).replace(".zip", "")
        plugin_name, plugin_version = plugin_basename.rsplit("-", 1)
        plugin_dir = os.path.join(EXTPLUGINS_DIR, plugin_name)
        if update and os.path.isdir(plugin_dir):
            shutil.rmtree(plugin_dir)
        if not os.path.isdir(plugin_dir):
            with zipfile.ZipFile(plugin, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(plugin_dir))
        else:
            status = 1
        os.remove(plugin)
    # Add a plugin from a path
    else:
        plugin_name = os.path.basename(location)
        plugin_version = None
        plugin_dir = os.path.join(EXTPLUGINS_DIR, plugin_name)
        print(plugin_dir)
        if not os.path.isdir(plugin_dir):
            os.symlink(location, plugin_dir)
        else:
            status = 1

    # Get the plugin information
    plugin_desc_file = glob.glob(
        os.path.join(plugin_dir, "*", "pysap-extplugin"))
    error_msg = (
        "'{0}' does not contain a valid plugin. A plugin is a Python "
        "module (ie with an __init__.py file that makes it importable) "
        "with at least a plugin description file named "
        "<myplugin>.pysap-extplugin where <myplugin> is the name of the "
        "Python module. The description file must contains the following "
        "keys: 'provide', 'author', 'version', 'description'".format(location))
    if len(plugin_desc_file) != 1:
        raise ValueError(error_msg)
    plugin_mod_dir = os.path.dirname(plugin_desc_file[0])
    plugin_mod = os.path.basename(plugin_mod_dir)
    with open(plugin_desc_file[0], "rt") as open_file:
        plugin_info = json.load(open_file)

    # Check that required attributes are specified
    for key in ("provide", "author", "version", "description"):
        if key not in plugin_info:
            raise ValueError(error_msg)

    # Check that we are dealing with a Python module
    plugin_mod_init = os.path.join(plugin_mod_dir, "__init__.py")
    if not os.path.isfile(plugin_mod_init):
        raise ValueError(error_msg)

    # Check the plugin is not already installed
    error_msg = (
        "Module '{0}' is already installed. First uninstall it using the "
        "'removeplugin' function or pip.".format(plugin_mod))
    try:
        mod = importlib.import_module(plugin_mod)
    except:
        mod = None
    if mod is not None:
        raise ValueError(
            "Module '{0}' is already installed. First uninstall it using the "
            "'removeplugin' function or pip.".format(plugin_mod))
    pysap_link = os.path.join(pysap_plugin_dir, plugin_mod)
    if os.path.isdir(pysap_link):
        raise ValueError(
            "A plugin with the same name is already installed.".format(
                plugin_mod))

    # Install the plugin in development mode
    # ToDo: use pip -e instead
    cwd = os.getcwd()
    os.chdir(plugin_dir)
    cmd = ["python", "setup.py", "develop", "--user"]
    subprocess.check_call(cmd)
    os.chdir(cwd)

    # Link the plugin
    if not os.path.isdir(pysap_link):
        os.symlink(plugin_mod_dir, pysap_link)

    # Update the plugins factory
    with open(EXTPLUGINS_FILE, "rt") as open_file:
        extplugins = json.load(open_file)
    plugin_info["location"] = plugin_dir
    extplugins[plugin_mod] = plugin_info
    with open(EXTPLUGINS_FILE, "wt") as open_file:
        json.dump(extplugins, open_file, indent=4)

    return status


def removeplugin(name):
    """ Remove an existing plugin.

    Parameters
    ----------
    name: str
        the name of the plugin to be removed.

    Returns
    -------
    status: int
        the command status: 0 removed, 1 already removed.
    """
    # Check for the reuested plugin in the registery
    with open(EXTPLUGINS_FILE, "rt") as open_file:
        extplugins = json.load(open_file)
    if name not in extplugins:
        print("No plugin registered with name '{0}'.".format(name))
        return 1
    plugin_info = extplugins[name]

    # Check that the pysap plugins directory is in write mode
    pysap_plugin_dir = os.path.dirname(__file__)
    if not os.access(pysap_plugin_dir, os.W_OK):
        raise ValueError(
            "For the moment the external plugin mechanism only works with "
            "user installation. Please use the 'pip install --user' directive "
            "to install pysap.")

    # Uninstall the plugin
    plugin_dir = plugin_info["location"]
    cwd = os.getcwd()
    os.chdir(plugin_dir)
    cmd = ["python", "setup.py", "develop", "--user", "--uninstall"]
    subprocess.check_call(cmd)
    os.chdir(cwd)

    # Remove the link
    pysap_link = os.path.join(pysap_plugin_dir, name)
    if os.path.islink(pysap_link):
        os.unlink(pysap_link)

    # Update the plugins factory
    del extplugins[name]
    with open(EXTPLUGINS_FILE, "wt") as open_file:
        json.dump(extplugins, open_file, indent=4)

    return 0
