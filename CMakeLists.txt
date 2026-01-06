cmake_minimum_required(VERSION 3.20)
project(POGS VERSION 0.4.0 LANGUAGES CXX)

# Set C++ standard
# Phase 2: Now using C++20 for modernization
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(POGS_BUILD_GPU "Build GPU support" ON)
option(POGS_BUILD_TESTS "Build tests" ON)
option(POGS_BUILD_EXAMPLES "Build examples" ON)
option(POGS_BUILD_PYTHON "Build Python bindings" OFF)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find dependencies
option(POGS_USE_SCIPY_OPENBLAS "Use scipy-openblas32 (Windows)" OFF)

if(POGS_USE_SCIPY_OPENBLAS)
  # Windows: Use scipy-openblas32 which provides MSVC-compatible OpenBLAS
  # scipy-openblas32 exports prefixed symbols (scipy_cblas_*, scipy_*_)
  # We use force-include of openblas_prefix.h to remap at compile time
  find_package(Python COMPONENTS Interpreter REQUIRED)

  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_lib_dir())"
    OUTPUT_VARIABLE SCIPY_OPENBLAS_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SCIPY_OPENBLAS_RESULT
  )
  if(NOT SCIPY_OPENBLAS_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find scipy-openblas32. Install with: pip install scipy-openblas32")
  endif()

  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_include_dir())"
    OUTPUT_VARIABLE SCIPY_OPENBLAS_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(SCIPY_OPENBLAS_LIB_FILE "${SCIPY_OPENBLAS_LIB_DIR}/libscipy_openblas.lib")
  set(SCIPY_OPENBLAS_DLL_FILE "${SCIPY_OPENBLAS_LIB_DIR}/libscipy_openblas.dll")

  if(NOT EXISTS "${SCIPY_OPENBLAS_LIB_FILE}")
    message(FATAL_ERROR "scipy-openblas32 library not found at: ${SCIPY_OPENBLAS_LIB_FILE}")
  endif()

  message(STATUS "Found scipy-openblas32:")
  message(STATUS "  Library: ${SCIPY_OPENBLAS_LIB_FILE}")
  message(STATUS "  DLL: ${SCIPY_OPENBLAS_DLL_FILE}")
  message(STATUS "  Include: ${SCIPY_OPENBLAS_INCLUDE_DIR}")

  # Store paths for direct linking in src/CMakeLists.txt
  # (Using direct linking instead of IMPORTED target for better MSVC compatibility)
  set(POGS_SCIPY_OPENBLAS_LIB "${SCIPY_OPENBLAS_LIB_FILE}" CACHE INTERNAL "scipy-openblas32 import library")
  set(POGS_SCIPY_OPENBLAS_INCLUDE "${SCIPY_OPENBLAS_INCLUDE_DIR}" CACHE INTERNAL "scipy-openblas32 include dir")
  set(POGS_OPENBLAS_PREFIX_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/include/openblas_prefix.h"
      CACHE INTERNAL "Path to OpenBLAS symbol prefix header")

else()
  # Unix/macOS: Use system BLAS/LAPACK
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

# GPU support
if(POGS_BUILD_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Building with GPU support (CUDA found)")
else()
  message(STATUS "Building CPU-only version")
endif()

# Platform-specific settings
if(APPLE)
  # Use Accelerate framework on macOS
  find_library(ACCELERATE_LIBRARY Accelerate)
  if(ACCELERATE_LIBRARY)
    message(STATUS "Found Accelerate framework: ${ACCELERATE_LIBRARY}")
  endif()
elseif(UNIX)
  # Use OpenBLAS on Linux if available
  find_package(OpenBLAS QUIET)
  if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS")
  endif()
endif()

# Setup installation directories (must be before add_subdirectory)
include(GNUInstallDirs)

# Add subdirectories
add_subdirectory(src)

if(POGS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(POGS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Installation
# Install headers (when we create the new include/ directory)
# install(DIRECTORY include/pogs
#         DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Export targets (for find_package support)
install(EXPORT POGSTargets
        FILE POGSTargets.cmake
        NAMESPACE pogs::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/POGS)

# Create package config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/POGSConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/POGSConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/POGSConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/POGS
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/POGSConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/POGSConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/POGS
)

# Print configuration summary
message(STATUS "")
message(STATUS "POGS Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU support: ${POGS_BUILD_GPU}")
message(STATUS "  Build tests: ${POGS_BUILD_TESTS}")
message(STATUS "  Build examples: ${POGS_BUILD_EXAMPLES}")
message(STATUS "")
