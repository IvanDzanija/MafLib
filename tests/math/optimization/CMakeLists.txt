# tests/math/optimization/CMakeLists.txt
set(TEST_NAME OptimizationTests)

add_executable(${TEST_NAME}
    OptimizationTests.cpp
)

target_link_libraries(${TEST_NAME}
    PRIVATE
        MafLib::MafLib
)

target_include_directories(${TEST_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/tests
)

add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
