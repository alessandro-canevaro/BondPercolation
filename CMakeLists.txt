cmake_minimum_required (VERSION 2.8)
project("BondPercolation")

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(net source/networklib.cpp include/networklib.h)
add_library(deg source/degreedist.cpp include/degreedist.h)
add_library(feat source/featuredist.cpp include/featuredist.h)
add_library(perc source/percolationlib.cpp include/percolationlib.h)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_executable (main source/main.cpp)

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#target_compile_options(main PUBLIC -fopenmp)
#target_compile_options(main PUBLIC -O3)
#target_compile_options(main PUBLIC -std=c++11)
#SET( CMAKE_CXX_FLAGS "-std=c++11 -O3 -fopenmp")

target_link_libraries(main "${OpenMP_CXX_FLAGS}")

#target_link_libraries(main OpenMP::OpenMP_CXX)
target_link_libraries(main net)
target_link_libraries(main deg)
target_link_libraries(main feat)
target_link_libraries(main perc)
target_link_libraries(perc net)
target_link_libraries(perc feat)

#cmake --build .\build\ --config Release
