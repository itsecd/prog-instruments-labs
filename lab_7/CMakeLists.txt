cmake_minimum_required(VERSION 3.20)

project("lr3" CXX)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME} "main.cpp" "sorts.h" "LinkedList.h")

target_compile_options(${PROJECT_NAME} PRIVATE "-Wall" "-Weffc++" "-Wextra" "-Wpedantic")

add_executable(profile_sorts "profile_sorts.cpp" "sorts.h" "LinkedList.h")
target_compile_options(profile_sorts PRIVATE "-Wall" "-Weffc++" "-Wextra" "-Wpedantic")
