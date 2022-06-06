cmake_minimum_required (VERSION 2.9)
project("BondPercolation")

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(net source/networklib.cpp include/networklib.h)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/source)

add_executable (main source/main.cpp)
target_link_libraries(main net)

#cmake --build .\build\ --config Release