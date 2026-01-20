cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(smpl_lbs_test)

# Set C++17
set(CMAKE_CXX_STANDARD 17)
 
list(APPEND CMAKE_PREFIX_PATH "D:/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
 
add_executable(smpl_test main.cpp)
target_link_libraries(smpl_test "${TORCH_LIBRARIES}")
  