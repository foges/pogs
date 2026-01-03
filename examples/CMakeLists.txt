# POGS Examples CMake Configuration

# Helper function to create example executables
function(add_pogs_example name source_file)
    add_executable(${name} ${source_file})
    target_link_libraries(${name} PRIVATE pogs::cpu)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/cpu/include
    )
endfunction()

# C++ Examples
# Note: Many example files are library functions included in run_all.cpp
# and don't have their own main() function. For Phase 1, we only build
# examples that work with the current library state.

# Cone examples that build successfully
add_pogs_example(test_sdp cpp_cone/test_sdp.cpp)

# Anderson acceleration benchmark (now implemented!)
add_pogs_example(anderson_benchmark cpp/anderson_benchmark.cpp)

# Excluded examples - to be fixed in Phase 2 modernization:
# add_pogs_example(run_all cpp/run_all.cpp)  # Needs example libs
# add_pogs_example(test_qr cpp/test_qr.cpp)  # Uses QR functions not in library
# add_pogs_example(lp_eq_cone cpp_cone/lp_eq.cpp)  # No main() function
# add_pogs_example(lasso_anderson cpp/lasso_anderson.cpp)  # Needs update

# C interface example - excluded for now (needs main())
# add_executable(test_c_interface cpp_cone/test_c_interface.c)
# target_link_libraries(test_c_interface PRIVATE pogs::cpu)
# target_include_directories(test_c_interface PRIVATE
#     ${CMAKE_SOURCE_DIR}/src/include
#     ${CMAKE_SOURCE_DIR}/src/interface_c
# )

# Note: More examples will be added during Phase 2 modernization
