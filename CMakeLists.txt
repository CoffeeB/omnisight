cmake_minimum_required(VERSION 3.20)
project(omnisight_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler optimization flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Include directories
include_directories(cpp/include)

# Core C++20 Geometry and Acceleration Library
add_library(omnisight_core STATIC
    cpp/src/geometry.cpp
    cpp/src/fast_nms.cpp
)

# pybind11 integration (optional if installed)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    message(STATUS "Found pybind11 v${pybind11_VERSION}. Building Python bindings...")
    pybind11_add_module(omnisight_accel
        cpp/bindings.cpp
    )
    target_link_libraries(omnisight_accel PRIVATE omnisight_core)
else()
    message(STATUS "pybind11 not found in CMake search path. Python C++ bindings skipped (pure C++ core will still build).")
endif()

# Standalone C++ test executable
add_executable(test_geometry_cpp
    cpp/src/test_main.cpp
)
target_link_libraries(test_geometry_cpp PRIVATE omnisight_core)
