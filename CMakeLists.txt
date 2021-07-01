cmake_minimum_required(VERSION 3.13)
project(SLAM)

add_executable(ORB_Feature ${CMAKE_CURRENT_SOURCE_DIR}/src/orb_feature.cpp)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(dependencies
        OpenCV)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ORB_Feature ${OpenCV_LIBS})
