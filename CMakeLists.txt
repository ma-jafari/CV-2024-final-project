cmake_minimum_required(VERSION 3.13)

# Set the project name
project(BilliardAnalysis)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O2 -g -Wall")

# Check the platform and adjust settings for Windows
if (WIN32)
    # On Windows, ensure OpenCV is properly located
    set(OpenCV_DIR "E:\\software\\programming\\opencv\\opencv\\build") # Adjust this path to your actual OpenCV build directory
endif()

# OpenCV package
find_package(OpenCV REQUIRED)

# Add the include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(BilliardAnalysis ${SOURCES})

# Link OpenCV
target_link_libraries(BilliardAnalysis ${OpenCV_LIBS})

# Print the OpenCV include directories (for debugging purposes)
message("OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
