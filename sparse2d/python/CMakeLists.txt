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
  include(BuildCfitsIO) # Custom module
  include(BuildSparse2D) # Custom module

  # Includes
  include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
  include_directories(${cfitsio_INCLUDE_DIR})
  include_directories(${sparse2d_INCLUDE_DIR})
  link_directories(${Boost_LIBRARY_DIR})
  link_directories(${cfitsio_LIBRARY_DIR})
  link_directories(${sparse2d_LIBRARY_DIR})

  # Build and link the pylib module
  add_library(pysparse SHARED pysparse.cpp)
  target_link_libraries(pysparse ${sparse2d_LIBRARIES} ${cfitsio_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  add_dependencies(pysparse Boost)
  add_dependencies(pysparse cfitsio)
  add_dependencies(pysparse sparse2d)

  # Tweaks the name of the library to match what Python expects
  set_target_properties(pysparse PROPERTIES SUFFIX .so)
  set_target_properties(pysparse PROPERTIES PREFIX "")
