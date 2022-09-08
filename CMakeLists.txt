cmake_minimum_required (VERSION 2.9)
project("BondPercolation")

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(net source/networklib.cpp include/networklib.h)
add_library(deg source/degreedist.cpp include/degreedist.h)
add_library(feat source/featuredist.cpp include/featuredist.h)
add_library(perc source/percolationlib.cpp include/percolationlib.h)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/source)

add_executable (main source/main.cpp)
target_link_libraries(main net)
target_link_libraries(main deg)
target_link_libraries(main feat)
target_link_libraries(main perc)
target_link_libraries(perc net)
target_link_libraries(perc feat)

#cmake --build .\build\ --config Release