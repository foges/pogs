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
option(POGS_USE_SCIPY_OPENBLAS "Use scipy-openblas32 for BLAS (Windows)" OFF)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find dependencies
# On Windows, optionally use scipy-openblas32 for MSVC-compatible BLAS
if(WIN32 AND POGS_USE_SCIPY_OPENBLAS)
  # Find scipy-openblas32 via Python
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_lib_dir())"
    OUTPUT_VARIABLE SCIPY_OPENBLAS_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SCIPY_OPENBLAS_RESULT
  )
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_include_dir())"
    OUTPUT_VARIABLE SCIPY_OPENBLAS_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(SCIPY_OPENBLAS_RESULT EQUAL 0)
    message(STATUS "Found scipy-openblas32:")
    message(STATUS "  Library dir: ${SCIPY_OPENBLAS_LIB_DIR}")
    message(STATUS "  Include dir: ${SCIPY_OPENBLAS_INCLUDE_DIR}")
    # Set BLAS and LAPACK to use scipy-openblas
    set(BLAS_LIBRARIES "${SCIPY_OPENBLAS_LIB_DIR}/scipy_openblas.lib")
    set(LAPACK_LIBRARIES "${SCIPY_OPENBLAS_LIB_DIR}/scipy_openblas.lib")
    set(BLAS_FOUND TRUE)
    set(LAPACK_FOUND TRUE)
    include_directories(${SCIPY_OPENBLAS_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "scipy-openblas32 not found. Install with: pip install scipy-openblas32")
  endif()
else()
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
