cmake_minimum_required(VERSION 3.22)
project(Nanobind_Eigensolver)

set(MADNESS_DIR "../madness/lib/cmake/madness")
find_package(MADNESS CONFIG REQUIRED)

set(DEV_MODULE Development.Module)

find_package(Python 3.11 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(my_ext my_ext.cpp)
# target_link_libraries(eigensolver PRIVATE madness)


