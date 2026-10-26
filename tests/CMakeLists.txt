# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

message(STATUS "Building tests for MafLib")

# Add subfolders for categories
add_subdirectory(math)

# Make the current directory visible for includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
