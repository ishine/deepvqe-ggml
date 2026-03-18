cmake_minimum_required(VERSION 3.20)
project(deepvqe-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ggml from git submodule (ggml/vendor/ggml)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory(vendor/ggml)

# Common library (npy I/O, comparison utilities)
add_library(dvqe_common STATIC common.cpp)
target_include_directories(dvqe_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dvqe_common PUBLIC ggml)

# Main inference binary
add_executable(deepvqe deepvqe.cpp)
target_link_libraries(deepvqe PRIVATE dvqe_common)

# Per-block test binaries (only built if source exists)
foreach(block fe encoder bottleneck decoder ccm align)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_${block}.cpp")
        add_executable(test_${block} test_${block}.cpp)
        target_link_libraries(test_${block} PRIVATE dvqe_common)
    endif()
endforeach()
