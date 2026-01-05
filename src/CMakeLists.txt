# POGS Library CMake Configuration

# CPU source files
set(POGS_CPU_SOURCES
    cpu/pogs.cpp
    cpu/matrix/matrix_dense.cpp
    cpu/matrix/matrix_sparse.cpp
    cpu/projector/projector_cgls.cpp
    cpu/projector/projector_direct_dense.cpp
)

# C interface sources
set(POGS_C_INTERFACE_SOURCES
    interface_c/pogs_c.cpp
)

# GPU source files (if enabled)
if(POGS_BUILD_GPU)
    set(POGS_GPU_SOURCES
        gpu/pogs.cu
        gpu/matrix/matrix_dense.cu
        gpu/matrix/matrix_sparse.cu
        gpu/projector/projector_cgls.cu
        gpu/projector/projector_direct_dense.cu
    )
endif()

# Create CPU library
add_library(pogs_cpu STATIC
    ${POGS_CPU_SOURCES}
    ${POGS_C_INTERFACE_SOURCES}
)

# Set include directories for CPU library
target_include_directories(pogs_cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies for CPU library
if(POGS_BLAS_LIBRARIES)
    # On Windows with scipy-openblas32, use full path directly
    message(STATUS "POGS CPU: Linking with ${POGS_BLAS_LIBRARIES}")
    # Convert to native path for Windows
    file(TO_CMAKE_PATH "${POGS_BLAS_LIBRARIES}" POGS_BLAS_LIB_CMAKE)
    target_link_libraries(pogs_cpu PUBLIC "${POGS_BLAS_LIB_CMAKE}")
    if(POGS_BLAS_INCLUDE_DIRS)
        target_include_directories(pogs_cpu PUBLIC "${POGS_BLAS_INCLUDE_DIRS}")
    endif()
else()
    # On Linux/macOS, use standard BLAS/LAPACK targets
    target_link_libraries(pogs_cpu PUBLIC BLAS::BLAS LAPACK::LAPACK)
endif()

# Platform-specific libraries
if(APPLE)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(pogs_cpu PUBLIC ${ACCELERATE_LIBRARY})
    endif()
endif()

# Set compiler warnings
target_compile_options(pogs_cpu
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Create alias for easier usage
add_library(pogs::cpu ALIAS pogs_cpu)

# Create shared library for Python bindings
add_library(pogs_cpu_shared SHARED
    ${POGS_CPU_SOURCES}
    ${POGS_C_INTERFACE_SOURCES}
)

target_include_directories(pogs_cpu_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies for shared library
if(POGS_BLAS_LIBRARIES)
    # On Windows with scipy-openblas32, use full path directly
    message(STATUS "POGS CPU Shared: Linking with ${POGS_BLAS_LIBRARIES}")
    target_link_libraries(pogs_cpu_shared PUBLIC "${POGS_BLAS_LIB_CMAKE}")
    if(POGS_BLAS_INCLUDE_DIRS)
        target_include_directories(pogs_cpu_shared PUBLIC "${POGS_BLAS_INCLUDE_DIRS}")
    endif()
else()
    # On Linux/macOS, use standard BLAS/LAPACK targets
    target_link_libraries(pogs_cpu_shared PUBLIC BLAS::BLAS LAPACK::LAPACK)
endif()

if(APPLE)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(pogs_cpu_shared PUBLIC ${ACCELERATE_LIBRARY})
    endif()
endif()

# Set output name to match what Python bindings expect
set_target_properties(pogs_cpu_shared PROPERTIES
    OUTPUT_NAME pogs_cpu
)

# GPU library (optional)
if(POGS_BUILD_GPU)
    add_library(pogs_gpu STATIC
        ${POGS_GPU_SOURCES}
    )

    # Set include directories for GPU library
    target_include_directories(pogs_gpu
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Link CUDA libraries
    target_link_libraries(pogs_gpu
        PUBLIC
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
    )

    # Set CUDA architectures (adjust as needed)
    set_target_properties(pogs_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80"
    )

    # Create alias
    add_library(pogs::gpu ALIAS pogs_gpu)

    # Combined library that includes both CPU and GPU
    add_library(pogs INTERFACE)
    target_link_libraries(pogs INTERFACE pogs::cpu pogs::gpu)
    add_library(pogs::pogs ALIAS pogs)
else()
    # CPU-only build: pogs is just an alias to pogs_cpu
    add_library(pogs INTERFACE)
    target_link_libraries(pogs INTERFACE pogs::cpu)
    add_library(pogs::pogs ALIAS pogs)
endif()

# Installation
install(TARGETS pogs_cpu pogs_cpu_shared
        EXPORT POGSTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(POGS_BUILD_GPU)
    install(TARGETS pogs_gpu
            EXPORT POGSTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cpu/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

if(POGS_BUILD_GPU)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gpu/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.cuh")
endif()
