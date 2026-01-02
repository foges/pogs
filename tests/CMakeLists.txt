# POGS Tests CMake Configuration

# Fetch Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.1
)
FetchContent_MakeAvailable(Catch2)

# Add Catch2 to module path for helper functions
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# Modern Catch2-based tests
add_executable(pogs_tests
    test_main.cpp
    # test_types.cpp        # TODO: Enable when modern API is complete
    # test_proximal.cpp     # TODO: Enable when modern API is complete
    test_cone_projections.cpp
    test_solver.cpp
)

target_link_libraries(pogs_tests
    PRIVATE
        pogs::cpu
        Catch2::Catch2WithMain
)

target_include_directories(pogs_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/cpu/include
)

target_compile_features(pogs_tests PRIVATE cxx_std_20)

# Discover tests
catch_discover_tests(pogs_tests)

# Legacy test (keep for now)
add_executable(test_cone_legacy test_cone.cpp)

target_link_libraries(test_cone_legacy
    PRIVATE
        pogs::cpu
)

target_include_directories(test_cone_legacy
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME test_cone_legacy COMMAND test_cone_legacy)
