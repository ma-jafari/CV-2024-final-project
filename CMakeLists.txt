cmake_minimum_required(VERSION 3.13)

# Set the project name
project(BilliardAnalysis)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory
include_directories(include)

# Add the executable
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(BilliardAnalysis ${SOURCES})

