cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

option(DEBUG_MODE_ENC "debug enc computation result" OFF)
option(DEBUG_MODE_DEC "debug dec computation result" OFF)
option(DEBUG_RESULT_ATTENTION "debug enc attention computation result" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

# Require C++14 and disable compiler-specific extensions
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DEFAULT_BUILD_TYPE "Release")

#-------------------------------------------------------------------------------
# Project name and version
#-------------------------------------------------------------------------------
project(FMHAInfer LANGUAGES C CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.0.0)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR
            "FMHAInfer does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})


set(SRC_FILES_DIR ${PROJECT_SOURCE_DIR}/src)
set(HDR_FILES_DIR ${PROJECT_SOURCE_DIR}/inc)
set(FMHA_HDR_FILES_DIR ${PROJECT_SOURCE_DIR}/fmha)
# Enable languages.
enable_language(CXX C)

# Use ${CMAKE_CXX_COMPILER} as the cuda host compiler.
if (NOT CMAKE_CUDA_HOST_COMPILER)
	set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

### Find Required Libs ###
find_package(CUDA 11.6 REQUIRED)
find_package(CUDAToolkit)

# Find CUDA ARCH VERSION
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET( ${CUDA_ARCH_LIST})
list (GET CUDA_ARCH_LIST 0 ARCH_NUMBER)
SET(CUDA_ARCH sm_${ARCH_NUMBER})
SET(CUDA_COMPUTE compute_${ARCH_NUMBER})


message(--------------------------------------------)
message(CUDA_ARCH = ${CUDA_ARCH})
message(CUDA_COMPUTE = ${CUDA_COMPUTE})
message(--------------------------------------------)



set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(CUDA_HEADER_DIRS ${CUDA_PATH}/include)
set(CUDA_LIB_DIRS ${CUDA_PATH}/lib64)


set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

set(CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr --keep --verbose --compiler-options -fPIC -g -v -G -Xcompiler -Wall")
set(CUDA_FLAGS "-gencode code=${CUDA_ARCH},arch=${CUDA_COMPUTE} ${CUDA_FLAGS}")
string(REGEX REPLACE "^sm_" "" ARCH_NUMBER "${CUDA_ARCH}")

if ("${CUDA_ARCH}" STREQUAL "sm_${ARCH_NUMBER}")
	set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUMBER}")
else()
	message(FATAL_ERROR "Unknown CUDA_ARCH: ${CUDA_ARCH}")
endif()

message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS}" CACHE STRING "CUDA flags set for FMHAInfer" FORCE)

#-------------------------------------------------------------------------------
# Process configuration options
#-------------------------------------------------------------------------------
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -lpthread")


message(STATUS "FMHAInfer build type: CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "FMHAInfer version: v${FMHAInfer_VERSION_STRING}")

# Headers and sources
set(SOURCES 
      ${SRC_FILES_DIR}/fmha_i8.cu
      )
set(HEADERS 
      ${HDR_FILES_DIR}/cpuGEMM.hpp
      ${HDR_FILES_DIR}/cpuSoftmax.hpp
      ${HDR_FILES_DIR}/fmha_i8.cuh
      ${HDR_FILES_DIR}/fmha_i8.h
      ${HDR_FILES_DIR}/fmha_param_i8.h
      ${HDR_FILES_DIR}/utils.hpp
      )

set(MAIN_FILE ${PROJECT_SOURCE_DIR}/test_fmha_i8.cpp)
# Declaring the library
set(ALL_SRC ${HEADERS} ${MAIN_FILE} ${SOURCES})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()


add_executable(FMHAInfer ${ALL_SRC})
target_include_directories(FMHAInfer PRIVATE ${CUDA_HEADER_DIRS} SYSTEM ${CUDA_INCLUDE_DIRS} ${FMHA_HDR_FILES_DIR} ${HDR_FILES_DIR})
target_link_directories(FMHAInfer PRIVATE ${CUDA_LIB_DIRS})
target_link_libraries (FMHAInfer PRIVATE ${CUDA_LIBRARIES} )
set_target_properties(FMHAInfer PROPERTIES POSITION_INDEPENDENT_CODE ON)