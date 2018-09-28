cmake_minimum_required(VERSION 3.12.0)
project(_pyNNGP CXX)
set(CMAKE_VERBOSE_MAKEFILE True)
set(CMAKE_BUILD_TYPE Release)
cmake_policy(SET CMP0063 NEW)  # Compile the static lib with hidden visibility.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include eigen)
add_subdirectory(pybind11)
set(PYSRC_FILES
  pysrc/pyNNGP.cpp
  pysrc/SeqNNGP.cpp
  pysrc/CovModel.cpp
)
set(SRC_FILES
  src/SeqNNGP.cpp
  src/tree.cpp
  src/utils.cpp
)
set(CMAKE_POSITION_INDEPENDENT_CODE True)

# Start OPENMP Support
# Really need a way to figure out how to set these without hard-coding them...
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.a")
find_package(OpenMP REQUIRED)
if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP_TARGET INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
    target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
    find_package(Threads REQUIRED)
    target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
    target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
endif()

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
# End OPENMP Support

add_library(pyNNGP STATIC ${SRC_FILES})
pybind11_add_module(_pyNNGP ${PYSRC_FILES})
target_link_libraries(_pyNNGP PUBLIC pyNNGP)
