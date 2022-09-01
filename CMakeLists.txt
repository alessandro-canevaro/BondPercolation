cmake_minimum_required (VERSION 2.9)
project("BondPercolation")

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(net source/networklib_old.cpp include/networklib_old.h)
add_library(deg source/degreedist.cpp include/degreedist.h)
add_library(feat source/featuredist.cpp include/featuredist.h)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/source)

add_executable (main source/main.cpp)
target_link_libraries(main net)
target_link_libraries(main deg)
target_link_libraries(main feat)

#cmake --build .\build\ --config Release