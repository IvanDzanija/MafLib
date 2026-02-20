# tests/math/CMakeLists.txt

message(STATUS "Building math tests for MafLib")

# Add subfolders for categories
add_subdirectory(linalg)
add_subdirectory(optimization)

# Make the current directory visible for includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
