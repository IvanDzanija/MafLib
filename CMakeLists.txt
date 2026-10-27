cmake_minimum_required(VERSION 3.20)
project(MafLib
    VERSION 1.0.0
    DESCRIPTION "Experimental C++ framework"
    LANGUAGES CXX
)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "This project requires Clang (or AppleClang), but ${CMAKE_CXX_COMPILER_ID} was found.")
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

if(APPLE)
    # OpenMP via Homebrew
    find_program(HOMEBREW brew)
    if(HOMEBREW)
        execute_process(COMMAND ${HOMEBREW} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(EXISTS "${LIBOMP_PREFIX}")
             set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
             set(OpenMP_CXX_LIB_NAMES omp)
             set(OpenMP_omp_LIBRARY ${LIBOMP_PREFIX}/lib/libomp.dylib)
             message(STATUS "Found Homebrew libomp at: ${LIBOMP_PREFIX}")
        else()
             message(STATUS "libomp not found via Homebrew (brew install libomp to enable OpenMP).")
        endif()
    else()
        message(STATUS "Homebrew not found; skipping OpenMP detection.")
    endif()

    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found. Enabling optimizations.")
        target_link_libraries(${PROJECT_NAME}
            INTERFACE
                "${ACCELERATE_FRAMEWORK}"
        )
        target_compile_definitions(${PROJECT_NAME}
            INTERFACE
                ACCELERATE_AVAILABLE
                ACCELERATE_NEW_LAPACK
                ACCELERATE_LAPACK_ILP64
                BLAS_AVAILABLE
        )
    else()
        message(STATUS "Accelerate framework NOT found. Building without it.")
    endif()
endif()

if(UNIX AND NOT APPLE)
    # OpenBLAS (Linux)
    find_path(OPENBLAS_INCLUDE_DIR
        NAMES cblas.h
        PATH_SUFFIXES openblas
    )
    find_library(OPENBLAS_LIBRARY
        NAMES openblas
    )
    if(OPENBLAS_LIBRARY)
        message(STATUS "OpenBLAS found. Enabling optimizations.")
        if(OPENBLAS_INCLUDE_DIR)
            target_include_directories(${PROJECT_NAME}
                INTERFACE
                    ${OPENBLAS_INCLUDE_DIR}
            )
        else()
            message(STATUS "OpenBLAS library found but cblas.h was not found; proceeding without an explicit include dir.")
        endif()
        target_link_libraries(${PROJECT_NAME}
            INTERFACE
                ${OPENBLAS_LIBRARY}
        )
        target_compile_definitions(${PROJECT_NAME}
            INTERFACE
                OPENBLAS_AVAILABLE
                BLAS_AVAILABLE
        )
    else()
        message(STATUS "OpenBLAS NOT found. Building without it.")
    endif()
endif()

# 4. Standard Dependencies
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Enabling parallelism.")
    target_link_libraries(${PROJECT_NAME}
        INTERFACE
            OpenMP::OpenMP_CXX
    )
    target_compile_definitions(${PROJECT_NAME}
        INTERFACE
            MAFLIB_OPENMP_AVAILABLE
    )
else()
    message(STATUS "OpenMP NOT found. Building without it.")
endif()

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# 5. Clang Optimizations
target_compile_options(${PROJECT_NAME}
    INTERFACE
        -fcolor-diagnostics
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g -O0 -fsanitize=address -fsanitize=undefined>
        $<$<CONFIG:Release>:-O3 -ffast-math -march=native -funroll-loops -DNDEBUG>
)
target_link_options(${PROJECT_NAME}
    INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address -fsanitize=undefined>
)

# 6. Installation Logic
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Local performance measurements. Targets are EXCLUDE_FROM_ALL: they are never
# built by a plain `cmake --build`, never picked up by ctest, and are meant to
# be run explicitly from a Release build:
#   cmake --build --preset clang-release --target LinAlgMeasure
add_subdirectory(measure)
