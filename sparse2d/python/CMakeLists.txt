cmake_minimum_required(VERSION 3.0)
include(ExternalProject)
include(FindPkgConfig)

# Adding customized cmake module for external packages
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(pysparse)

  # Find default python libraries and interpreter
  find_package(PythonInterp REQUIRED)
  if(NOT DEFINED PYTHON_LIBRARIES OR NOT DEFINED PYTHON_INCLUDE_DIRS)
    find_package(PythonLibs REQUIRED)
  else()
    message(STATUS "Using Python Lib: ${PYTHON_LIBRARIES}")
    message(STATUS "Using Python Inc: ${PYTHON_INCLUDE_DIRS}")
  endif()
  include_directories(${PYTHON_INCLUDE_DIRS})
  link_directories(${PYTHON_LIBRARY_DIRS})

  # Include Pybind11 headers
  message(STATUS "Using Pybind11 Inc: ${PYBIND11_INCLUDE_DIR}")
  include_directories(${PYBIND11_INCLUDE_DIR})

  # Find OpenMP
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(OpenMP_INCLUDE_PATH "/usr/local/include")
    set(OpenMP_LIB_PATH "/usr/local/lib")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_CXX_LIBRARIES "${OpenMP_LIB_PATH}/libomp.dylib")
    include_directories(${OpenMP_INCLUDE_PATH})
    link_directories(${OpenMP_LIB_PATH})
  else()
    find_package(OpenMP REQUIRED)
  endif()

  # Flags for MAC OSX
  if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(APPLE_FLAGS "-DMACOS")
  else(APPLE)
    set(APPLE_FLAGS "")
  endif(APPLE)

  # Compilation flags
  set(CMAKE_CXX_FLAGS "-std=c++11 -DNO_DISP_IO -ggdb3 -fPIC -O2 -ffast-math -fomit-frame-pointer ${APPLE_FLAGS} ${OpenMP_CXX_FLAGS} -Wno-write-strings -DNDEBUG")

  # Custom modules
  include(BuildCfitsIO)
  get_filename_component(SPARSE2D_INSTALL_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
  message(STATUS "Binary output path : ${SPARSE2D_INSTALL_DIR}")
  include(BuildSparse2D)

  # Includes
  include_directories(${PYTHON_INCLUDE_DIRS})
  include_directories(${cfitsio_INCLUDE_DIR})
  include_directories(${sparse2d_INCLUDE_DIR})
  include_directories(${OpenMP_INCLUDE_PATH})
  link_directories(${cfitsio_LIBRARY_DIR})
  link_directories(${sparse2d_LIBRARY_DIR})


  set(EXAMPLES pysparse)

  # Build and link the module
  foreach(program ${EXAMPLES})

    add_library(${program} SHARED ${program}.cpp)

    if(APPLE)
      target_link_libraries(${program} ${OpenMP_CXX_LIBRARIES} ${sparse2d_LIBRARIES} ${cfitsio_LIBRARIES})
    else(APPLE)
      target_link_libraries(${program} ${OpenMP_CXX_LIBRARIES} ${sparse2d_LIBRARIES} ${cfitsio_LIBRARIES} ${PYTHON_LIBRARIES})
    endif(APPLE)

    add_dependencies(${program} cfitsio)
    add_dependencies(${program} sparse2d)

    set_target_properties(${program} PROPERTIES SUFFIX .so)
    set_target_properties(${program} PROPERTIES PREFIX "")

    if(APPLE)
      set_target_properties(${program} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif(APPLE)

  endforeach(program)
