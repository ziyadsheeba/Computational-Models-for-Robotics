cmake_minimum_required(VERSION 3.5)

# set name of the project
project(assignment1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)

# thirdparty code
add_subdirectory(ext)

# our code
add_subdirectory(src)
