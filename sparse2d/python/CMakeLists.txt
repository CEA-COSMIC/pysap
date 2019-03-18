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

  # Find default python libraries and interpreter
  find_package(PythonInterp REQUIRED)
  if(NOT DEFINED PYTHON_LIBRARIES OR NOT DEFINED PYTHON_INCLUDE_DIRS)
    find_package(PythonLibs REQUIRED)
  else()
    message(STATUS "Using Python Lib: ${PYTHON_LIBRARIES}")
    message(STATUS "Using Python Inc: ${PYTHON_INCLUDE_DIRS}")
  endif()
  find_package(OpenMP REQUIRED)

  # Flags for MAC OSX
  if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(APPLE_FLAGS "-DMACOS")
  else(APPLE)
    set(APPLE_FLAGS "")
  endif(APPLE)

  # Compilation flags
  #set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CXX_FLAGS "-std=c++11 -DNO_DISP_IO -ggdb3 -fPIC -O2 -ffast-math -fomit-frame-pointer ${APPLE_FLAGS} ${OpenMP_CXX_FLAGS} -Wno-write-strings -DNDEBUG")

  # Custom modules
  include(BuildBoost)
  include(BuildCfitsIO)
  include(BuildSparse2D)

  # Includes
  include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
  include_directories(${cfitsio_INCLUDE_DIR})
  include_directories(${sparse2d_INCLUDE_DIR})
  include_directories(${OpenMP_INCLUDE_PATH})
  link_directories(${Boost_LIBRARY_DIR})
  link_directories(${cfitsio_LIBRARY_DIR})
  link_directories(${sparse2d_LIBRARY_DIR})

  # Build and link the pylib module
  add_library(pysparse SHARED pysparse.cpp)
  if(APPLE)
    target_link_libraries(pysparse ${OpenMP_CXX_LIBRARIES} ${sparse2d_LIBRARIES} ${cfitsio_LIBRARIES} ${Boost_LIBRARIES})
  else(APPLE)
    target_link_libraries(pysparse ${OpenMP_CXX_LIBRARIES} ${sparse2d_LIBRARIES} ${cfitsio_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  endif(APPLE)
  add_dependencies(pysparse Boost)
  add_dependencies(pysparse cfitsio)
  add_dependencies(pysparse sparse2d)

  # Tweaks the name of the library to match what Python expects
  set_target_properties(pysparse PROPERTIES SUFFIX .so)
  set_target_properties(pysparse PROPERTIES PREFIX "")

  if(APPLE)
    set_target_properties(pysparse PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(APPLE)
