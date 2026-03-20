cmake_minimum_required(VERSION 3.20)
project(deepvqe-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DEEPVQE_BUILD_SHARED "Build shared library with C API" OFF)

# Need PIC for everything when building shared lib (ggml static lib gets linked in)
if(DEEPVQE_BUILD_SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

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

# Model library (shared between CLI and C API)
add_library(dvqe_model STATIC deepvqe_model.cpp)
target_link_libraries(dvqe_model PUBLIC dvqe_common)

# Main inference binary (CLI)
add_executable(deepvqe deepvqe.cpp)
target_link_libraries(deepvqe PRIVATE dvqe_model)

# Shared library with C API (for purego / dlopen)
if(DEEPVQE_BUILD_SHARED)
    add_library(deepvqe_shared SHARED deepvqe_api.cpp)
    target_link_libraries(deepvqe_shared PRIVATE dvqe_model)
    target_compile_definitions(deepvqe_shared PRIVATE DEEPVQE_BUILD)
    set_target_properties(deepvqe_shared PROPERTIES
        OUTPUT_NAME "deepvqe"
        VERSION 0.1.0
        SOVERSION 0
    )

    # Hide all symbols except those marked DEEPVQE_API
    if(NOT WIN32)
        target_compile_options(deepvqe_shared PRIVATE -fvisibility=hidden)
    endif()

    install(TARGETS deepvqe_shared LIBRARY DESTINATION lib)
    install(FILES deepvqe_api.h DESTINATION include)
endif()

# Per-block test binaries (only built if source exists)
foreach(block fe encoder bottleneck decoder ccm align)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_${block}.cpp")
        add_executable(test_${block} test_${block}.cpp)
        target_link_libraries(test_${block} PRIVATE dvqe_common)
    endif()
endforeach()
