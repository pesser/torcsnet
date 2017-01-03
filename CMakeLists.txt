cmake_minimum_required(VERSION 3.5)

project(chiptorcs2)
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})

add_executable(visualize visualize.cpp)
target_link_libraries(visualize ${Caffe_LIBRARIES})

add_executable(split split.cpp)
target_link_libraries(split ${Caffe_LIBRARIES})

add_executable(normalize normalize.cpp)
target_link_libraries(normalize ${Caffe_LIBRARIES})

configure_file(network_train.prototxt network_train.prototxt)
configure_file(network_solver.prototxt network_solver.prototxt)
