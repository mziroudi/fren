cmake_minimum_required(VERSION 3.14)
project(fren_capture CXX C)
set(CMAKE_CXX_STANDARD 17)

# llama.cpp as a subproject, CPU only, no tools/examples/tests/server.
set(LLAMA_STANDALONE OFF)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/llama.cpp llama.cpp)

add_executable(route_trace route_trace.cpp)
target_include_directories(route_trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/llama.cpp/vendor)
target_link_libraries(route_trace PRIVATE llama ggml)
