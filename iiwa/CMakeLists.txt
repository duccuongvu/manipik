cmake_minimum_required(VERSION 3.16)
project(iiwa_ik_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen3 — header-only linear algebra used by iiwa_ik.hpp.
find_package(Eigen3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    # Fall back to the common system include path (e.g. /usr/include/eigen3).
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense PATH_SUFFIXES eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found. Install it (e.g. `sudo apt install libeigen3-dev`).")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# Standalone C++ example that solves IK via the header-only library.
add_executable(iiwa_ik_example iiwa_ik_example.cpp)
target_link_libraries(iiwa_ik_example PRIVATE Eigen3::Eigen)
target_include_directories(iiwa_ik_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
