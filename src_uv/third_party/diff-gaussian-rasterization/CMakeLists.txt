cmake_minimum_required(VERSION 3.20)
project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 1. We need Torch here because 'rasterize_points.cu' uses torch::Tensor
find_package(Torch REQUIRED)

# 2. Define the Library
# We include the core implementation AND the bridge file (rasterize_points.cu)
add_library(GaussianRasterizer STATIC
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.cu
    cuda_rasterizer/rasterizer_impl.cu
    rasterize_points.cu  # <--- CRITICAL: This contains the function 'RasterizeGaussiansCUDA'
)

# 3. Include Paths
# Public: So your main.cpp can include "rasterize_points.h" (if you make one)
target_include_directories(GaussianRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Private: GLM is only needed inside the rasterizer
# Note: Check if your glm folder is at "3rdparty/glm" or inside the repo.
# Adjust "../glm" below to match where you put the 'glm' folder relative to this file.
target_include_directories(GaussianRasterizer PRIVATE third_party/glm)

# 4. Link Torch
target_link_libraries(GaussianRasterizer PUBLIC ${TORCH_LIBRARIES})

# 5. CUDA Architectures (Optimize for your GPU)
# "native" will auto-detect your GPU. Use "75;86" if building for others.
set_target_properties(GaussianRasterizer PROPERTIES CUDA_ARCHITECTURES "native")

# Fix for Windows min/max macros
target_compile_definitions(GaussianRasterizer PRIVATE NOMINMAX)