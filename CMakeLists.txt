cmake_minimum_required(VERSION 3.22)
project(Nanobind_TeqMad)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
if (NOT Python_EXECUTABLE)
  message(FATAL_ERROR "Python 3.8 not found in the Conda environment.")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(MADNESS_DIR "../madness/lib/cmake/madness")
find_package(MADNESS CONFIG REQUIRED)

nanobind_add_module(1dharmonic 1dharmonic.cpp)
target_link_libraries(1dharmonic PRIVATE madness)

