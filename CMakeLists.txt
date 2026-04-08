cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(GaussianAvatar LANGUAGES CXX)

option(BUILD_DATASET_PREP "Build the standalone dataset preparation pipeline" ON)
option(BUILD_LEGACY_SRC "Build the legacy src-based applications" OFF)

if(BUILD_DATASET_PREP)
    add_subdirectory(dataset_prep)
endif()

if(BUILD_LEGACY_SRC)
    add_subdirectory(src)
endif()

if(NOT BUILD_DATASET_PREP AND NOT BUILD_LEGACY_SRC)
    message(WARNING "Both BUILD_DATASET_PREP and BUILD_LEGACY_SRC are OFF. No targets will be generated.")
endif()
