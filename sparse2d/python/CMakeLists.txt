##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

cmake_minimum_required(VERSION 2.8)
include(ExternalProject)
include(FindPkgConfig)

# Adding customized cmake module for building boost
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(pysparse)

  # Find cfitsio
  pkg_check_modules(CFITSIO REQUIRED cfitsio)
  include_directories(${CFITSIO_INCLUDE_DIRS})
  link_directories(${CFITSIO_LIBRARY_DIRS})

  # Define the sparse2d include directories
  set(Spase2d_INSTALL_DIR ${CMAKE_SOURCE_DIR}/../build/local_install/usr)
  set(Spase2d_INCLUDE_DIR ${Spase2d_INSTALL_DIR}/include /home/ag239446/git/pisap/sparse2d/src/libtools)
  include_directories(${Spase2d_INCLUDE_DIR})

  # Define the sparse2d libraries
  set(Sparse2d_LIBRARIES mga2d sparse2d sparse1d tools)
  link_directories(${Spase2d_INSTALL_DIR}/lib)

  # Define boost options
  SET(Boost_USE_STATIC_LIBS ON)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME ON)

  # Compilation flags
  #set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CXX_FLAGS "-std=c++03  -DNO_DISP_IO -ggdb3 -fPIC -O2 -ffast-math -fomit-frame-pointer ${OpenMP_CXX_FLAGS} -Wno-write-strings -DNDEBUG")

  # Find default python libraries and interpreter
  find_package(PythonInterp REQUIRED)
  find_package(PythonLibs REQUIRED)
  find_package(OpenMP REQUIRED)
  include(BuildBoost) # Custom module

  # Includes
  include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIR})

  # Build and link the pylib module
  add_library(pysparse SHARED pysparse.cpp)
  target_link_libraries(pysparse ${Sparse2d_LIBRARIES} -lcfitsio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  add_dependencies(pysparse Boost)

  # Tweaks the name of the library to match what Python expects
  set_target_properties(pysparse PROPERTIES SUFFIX .so)
  set_target_properties(pysparse PROPERTIES PREFIX "")
