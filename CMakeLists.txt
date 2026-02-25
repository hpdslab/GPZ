cmake_minimum_required(VERSION 3.22)

# change based on the GPU arch
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Project name and languages
project(gpz LANGUAGES CXX CUDA VERSION 1.0.0 DESCRIPTION "Error-bounded GPU lossy compression particle library")


# Optional nvcomp support: pass -DNVCOMP_DIR=/path/to/nvcomp to enable
if(DEFINED NVCOMP_DIR)
    set(GPZ_USE_NVCOMP ON)
    message(STATUS "nvcomp ENABLED, NVCOMP_DIR: ${NVCOMP_DIR}")
else()
    set(GPZ_USE_NVCOMP OFF)
    message(STATUS "nvcomp DISABLED (pass -DNVCOMP_DIR=<path> to enable)")
endif()

if(NOT DEFINED REGISTER_LIMIT)
    set(REGISTER_LIMIT 168)
endif()

# Find required CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=${REGISTER_LIMIT}")
message(STATUS "Using --maxrregcount=${REGISTER_LIMIT}")

# Debug flags
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo ")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xptxas -v ")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -v")

# Add the source files
file(GLOB CPP_SOURCES "src/*.cpp")
file(GLOB CU_SOURCES "src/*.cu")

add_executable(gpz ${CPP_SOURCES} ${CU_SOURCES})

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/components
        ${CMAKE_CURRENT_SOURCE_DIR}/components/include
)

if(GPZ_USE_NVCOMP)
    find_library(NVCOMP_LIB nvcomp
        PATHS "${NVCOMP_DIR}/lib" "${NVCOMP_DIR}/lib64"
        NO_DEFAULT_PATH
    )
    if(NOT NVCOMP_LIB)
        message(FATAL_ERROR "Could not find libnvcomp in ${NVCOMP_DIR}/lib or ${NVCOMP_DIR}/lib64. "
                "Please check NVCOMP_DIR is correct.")
    endif()
    message(STATUS "Found nvcomp library: ${NVCOMP_LIB}")
    target_include_directories(gpz PRIVATE "${NVCOMP_DIR}/include")
    target_link_libraries(gpz PRIVATE ${NVCOMP_LIB} CUDA::cudart)
    target_compile_definitions(gpz PRIVATE GPZ_USE_NVCOMP)
else()
    target_link_libraries(gpz PRIVATE CUDA::cudart)
endif()

set_target_properties(gpz PROPERTIES OUTPUT_NAME gpz)
