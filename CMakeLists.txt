cmake_minimum_required(VERSION 3.16)
project(faer CXX)

set(CMAKE_CXX_STANDARD 20)

get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
  set(FAER_TOP_LEVEL FALSE)
else()
  set(FAER_TOP_LEVEL TRUE)
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
add_library(
  faer ${CMAKE_CURRENT_SOURCE_DIR}/src/null.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/cache.cpp
)
if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL
                                                "GNU")
)
  target_compile_options(faer PUBLIC -march=native)
endif()

target_include_directories(faer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(
  faer SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/simde
)

add_subdirectory(external/veg)
set(BUILD_TESTING OFF)
add_subdirectory(external/cpu_features)

target_link_libraries(faer PUBLIC veg cpu_features)

option(FAER_ENABLE_TESTING "enable tests" ${FAER_TOP_LEVEL})
option(FAER_ENABLE_BENCHMARKS "enable benchmarks" ${FAER_TOP_LEVEL})

if(FAER_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(FAER_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
endif()
