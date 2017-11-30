#========================================================#
# Build the CfitsIO dependencies for the project         #
#========================================================#

set(cfitsioVersion 3.410)
set(cfitsioSHA256 a556ac7ea1965545dcb4d41cfef8e4915eeb8c0faa1b52f7ff70870f8bb5734c)

string(REPLACE "." "" cfitsioFolderName ${cfitsioVersion})

ExternalProject_Add(cfitsio
    PREFIX cfitsio
    URL  http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio${cfitsioFolderName}.tar.gz
    URL_HASH  SHA256=${cfitsioSHA256}
    CONFIGURE_COMMAND ./configure
        --prefix=${CMAKE_BINARY_DIR}/extern
    BUILD_COMMAND make install
        -j8
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    )

set(cfitsio_LIBRARY_DIR ${CMAKE_BINARY_DIR}/extern/lib/ )
set(cfitsio_INCLUDE_DIR ${CMAKE_BINARY_DIR}/extern/include/ )
set(cfitsio_LIBRARIES -lcfitsio)

