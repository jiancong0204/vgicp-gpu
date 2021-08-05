cmake_minimum_required(VERSION 3.16)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cuda_tutorial)

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

cmake_policy(SET CMP0104 NEW)

if(USE_CUDA)
    enable_language("CUDA")
    add_definitions(-DUSE_CUDA)
    find_package(CUDA  REQUIRED)
    include_directories("${CUDA_INCLUDE_DIRS}")
    add_executable(cuda_tutorial ${cpu_source_files} ${gpu_source_files})
else(USE_CUDA)
    add_executable(cuda_tutorial ${cpu_source_files})
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")