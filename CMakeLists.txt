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
  # Get the library name (returns just the name, not path)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32; print(scipy_openblas32.get_library())"
    OUTPUT_VARIABLE SCIPY_OPENBLAS_LIB_NAME
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # List files in lib directory for debugging
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import scipy_openblas32, os; d=scipy_openblas32.get_lib_dir(); print('Files in', d, ':', os.listdir(d) if os.path.isdir(d) else 'NOT A DIR')"
    OUTPUT_VARIABLE SCIPY_OPENBLAS_FILES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(SCIPY_OPENBLAS_RESULT EQUAL 0)
    message(STATUS "Found scipy-openblas32:")
    message(STATUS "  Library dir: ${SCIPY_OPENBLAS_LIB_DIR}")
    message(STATUS "  Include dir: ${SCIPY_OPENBLAS_INCLUDE_DIR}")
    message(STATUS "  Library name: ${SCIPY_OPENBLAS_LIB_NAME}")
    message(STATUS "  ${SCIPY_OPENBLAS_FILES}")

    # The .lib file is in the lib directory with the library name
    set(SCIPY_OPENBLAS_LIB_FILE "${SCIPY_OPENBLAS_LIB_DIR}/${SCIPY_OPENBLAS_LIB_NAME}.lib")
    set(SCIPY_OPENBLAS_DLL_FILE "${SCIPY_OPENBLAS_LIB_DIR}/${SCIPY_OPENBLAS_LIB_NAME}.dll")

    message(STATUS "  Lib file: ${SCIPY_OPENBLAS_LIB_FILE}")
    message(STATUS "  DLL file: ${SCIPY_OPENBLAS_DLL_FILE}")

    # Create BLAS::BLAS imported target for scipy-openblas
    # Use SHARED IMPORTED GLOBAL with IMPORTED_IMPLIB for Windows .lib files
    add_library(BLAS::BLAS SHARED IMPORTED GLOBAL)
    set_target_properties(BLAS::BLAS PROPERTIES
      IMPORTED_IMPLIB "${SCIPY_OPENBLAS_LIB_FILE}"
      IMPORTED_LOCATION "${SCIPY_OPENBLAS_DLL_FILE}"
      INTERFACE_INCLUDE_DIRECTORIES "${SCIPY_OPENBLAS_INCLUDE_DIR}"
    )
    # Create LAPACK::LAPACK imported target (same library)
    add_library(LAPACK::LAPACK SHARED IMPORTED GLOBAL)
    set_target_properties(LAPACK::LAPACK PROPERTIES
      IMPORTED_IMPLIB "${SCIPY_OPENBLAS_LIB_FILE}"
      IMPORTED_LOCATION "${SCIPY_OPENBLAS_DLL_FILE}"
      INTERFACE_INCLUDE_DIRECTORIES "${SCIPY_OPENBLAS_INCLUDE_DIR}"
    )
    set(BLAS_FOUND TRUE)
    set(LAPACK_FOUND TRUE)
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
