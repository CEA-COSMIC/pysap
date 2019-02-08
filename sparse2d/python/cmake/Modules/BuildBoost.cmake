#========================================================#
# Build the Boost dependencies for the project using a   #
#specific version of python                              #
#========================================================#

set(BoostVersion 1.68.0)
set(BoostSHA256 7f6130bc3cf65f56a618888ce9d5ea704fa10b462be126ad053e80e553d6d8b7)

string(REGEX REPLACE "beta\\.([0-9])$" "beta\\1" BoostFolderName ${BoostVersion})
string(REPLACE "." "_" BoostFolderName ${BoostFolderName})
set(BoostFolderName boost_${BoostFolderName})

ExternalProject_Add(Boost
    PREFIX Boost
    URL  https://dl.bintray.com/boostorg/release/${BoostVersion}/source/${BoostFolderName}.tar.bz2
    URL_HASH  SHA256=${BoostSHA256}
    CONFIGURE_COMMAND ./bootstrap.sh
        --with-libraries=python
        --with-python=${PYTHON_EXECUTABLE}
    BUILD_COMMAND ./b2 install
        variant=release
        link=static
        cxxflags='-fPIC'
        --prefix=${CMAKE_BINARY_DIR}/extern
        -d 0
        -j8
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    )

set(Boost_LIBRARY_DIR ${CMAKE_BINARY_DIR}/extern/lib/ )
set(Boost_INCLUDE_DIR ${CMAKE_BINARY_DIR}/extern/include/ )
set(PYTHON_EXT "${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
set(Boost_LIBRARIES "-lboost_python${PYTHON_EXT} -lboost_numpy${PYTHON_EXT}")
