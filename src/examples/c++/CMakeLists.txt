cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)

project(jetson-cam-capture-opencv)

#allow cmake to find opencv
find_package(OpenCV REQUIRED)

#Variable which include path to directories where header files are located
include_directories(${OpenCV_INCLUDE_DIRS})

#executable to create name and source file
add_executable(cam-capture jetson-cam-capture-opencv.cpp) 

#Variable which include path to open opencv libraries files
target_link_libraries(cam-capture ${OpenCV_LIBS})