#========================================================#
# Build the CfitsIO dependencies for the project         #
#========================================================#

set(sparse2dVersion 0.0.1)
set(sparse2dSHA256 baa03d78b00b061c95987d805c187599189c38570f5d11d86387c81de8409eb3)

ExternalProject_Add(sparse2d
    PREFIX sparse2d
    URL  $ENV{HOME}/git/pisap/sparse2d/sparse2d-${sparse2dVersion}.tar.gz
    URL_HASH  SHA256=${sparse2dSHA256}
    DEPENDS cfitsio
    CONFIGURE_COMMAND cmake
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extern
    BUILD_COMMAND make install
        -j8
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    )

set(sparse2d_LIBRARY_DIR ${CMAKE_BINARY_DIR}/extern/lib/ )
set(sparse2d_INCLUDE_DIR ${CMAKE_BINARY_DIR}/extern/include/ )
set(sparse2d_LIBRARIES -lmga2d -lsparse2d -lsparse1d -ltools)

