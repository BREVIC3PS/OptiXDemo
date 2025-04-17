cmake_minimum_required(VERSION 3.10)
project(OptiX9Example LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到 CUDA
find_package(CUDAToolkit REQUIRED)

# 设置 OptiX SDK 根目录（根据你机器上的路径修改）
set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
include_directories("${OPTIX_ROOT}/include")
# OptiX 9 使用 runtime 动态加载（lib 目录不用 link_directories，除非你手动构建 .lib）
# 一般使用 optixFunctionTable 的形式加载 optix API

# 添加第三方库（例如 tinyobjloader）
include_directories(${CMAKE_SOURCE_DIR}/third_party/tinyobjloader)

# 源文件列表（通配符不能直接用于 target，需使用 file(GLOB ...)）
file(GLOB_RECURSE SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.h"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(OptiX9Example ${SOURCES})

# 链接 CUDA runtime
target_link_libraries(OptiX9Example
    PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)

# 设置 CUDA 编译选项
set_target_properties(OptiX9Example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "86;89;80"
)

# MSVC 编译警告等级
if (MSVC)
    target_compile_options(OptiX9Example PRIVATE /W4)
endif()
