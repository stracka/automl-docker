cmake_minimum_required(VERSION 3.4...3.18)
#project(cmake_example)

#add_subdirectory(pybind11)
#pybind11_add_module(cmake_example src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(cmake_example
#                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


project(example)

find_package(ROOT 6.16 CONFIG REQUIRED)
find_package(pybind11 REQUIRED PATHS /home/foo/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/)

#add_subdirectory(pybind11)
add_executable(example main.cpp)
target_link_libraries(example PUBLIC pybind11::embed ROOT::Tree)
#target_link_libraries(RootSimpleExample PUBLIC 
