cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_compile_options(-freflection)
add_compile_options(-Wno-interference-size)

add_library(Matrix INTERFACE)
target_sources(Matrix INTERFACE Matrix.hpp)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE Matrix)


project(Matrix)