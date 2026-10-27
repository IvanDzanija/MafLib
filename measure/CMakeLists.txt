# measure/CMakeLists.txt
#
# Local performance measurements. The executables here are marked
# EXCLUDE_FROM_ALL, so:
#   - a plain `cmake --build` (e.g. on CI/remotes) never compiles them
#   - ctest never runs them
#   - they are still configured in the same generate step (one cmake call)
#
# Build and run locally (Release for meaningful numbers):
#   cmake --preset clang-release
#   cmake --build --preset clang-release --target LinAlgMeasure
#   ./build/clang-release/measure/LinAlgMeasure | tee measure/results/linalg.txt

set(MEASURE_NAME LinAlgMeasure)

add_executable(${MEASURE_NAME} EXCLUDE_FROM_ALL
    LinAlg.cpp
)

target_link_libraries(${MEASURE_NAME}
    PRIVATE
        MafLib::MafLib
)

target_include_directories(${MEASURE_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)
