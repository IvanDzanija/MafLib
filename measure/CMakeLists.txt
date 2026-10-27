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
