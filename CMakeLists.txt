cmake_minimum_required(VERSION 3.13)

# Set the project name
project(BilliardAnalysis)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O2 -g")

#OpenCV package
find_package( OpenCV REQUIRED )                              

# Add the include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(BilliardAnalysis ${SOURCES})

# Link OpenCV
target_link_libraries(BilliardAnalysis ${OpenCV_LIBS})

message("${OpenCV_INCLUDE_DIRS}")
