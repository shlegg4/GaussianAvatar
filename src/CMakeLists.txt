cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1. Enable C++ and CUDA support
project(smpl_lbs_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 2. Setup LibTorch
list(APPEND CMAKE_PREFIX_PATH "D:/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 3. Add the Rasterizer Subproject
# This looks into 'third_party/diff-gaussian-rasterization' and builds the library defined there
add_subdirectory(third_party/diff-gaussian-rasterization)

# 4. Define Your Main Executable
add_executable(smpl_test main.cpp)

# 5. Link Libraries
# We link Torch AND the new GaussianRasterizer library
target_link_libraries(smpl_test 
    PRIVATE 
    "${TORCH_LIBRARIES}" 
    GaussianRasterizer
)

# 6. Windows Specific Fixes
# Prevents conflicts between Windows.h min/max and std::min/max
target_compile_definitions(smpl_test PRIVATE NOMINMAX)

# Copy the smpl_data.pt file to the build directory (Optional convenience)
file(COPY ${CMAKE_SOURCE_DIR}/smpl_data.pt DESTINATION ${CMAKE_BINARY_DIR})