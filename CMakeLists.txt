cmake_minimum_required(VERSION 3.8)
project(NaiveBayes)

set(CMAKE_CXX_STANDARD 17)


file(GLOB_RECURSE sources src/*.cpp src/*.h)

add_executable(NaiveBayes
        ${sources})
