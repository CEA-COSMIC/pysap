############
# PySparse #
############

# -------------- #
# Project Set Up #
# -------------- #

cmake_minimum_required(VERSION 3.12.0)
project(pysparse)

  # Include custom modules in CMake module path
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

  # Load CMake tools
  include(ExternalProject)

  # Set install directory for Sparse2D binaries
  get_filename_component(SPARSE2D_INSTALL_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
  message(STATUS "Binary output path : ${SPARSE2D_INSTALL_DIR}")

  # Build Sparse2D
  include(BuildSparse2D)
