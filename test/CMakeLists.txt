include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/doctest.cmake)

add_library(faer_tests src/doctest_main.cpp)

add_library(project_warnings INTERFACE)
add_library(project_options INTERFACE)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/compiler_warnings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/static_analyzers.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/sanitizers.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/conan.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/extra_local_settings.cmake)

enable_sanitizers(project_options)
set_project_warnings(project_warnings)
target_link_libraries(faer PUBLIC project_options project_warnings)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/extra_local_settings.cmake)

include_directories(include)

# Conan dependencies
set(CONAN_REQUIRES
    fmt/8.1.1
    eigen/3.4.0
    backward-cpp/1.6
)
set(CONAN_OPTIONS)
run_conan()

target_link_libraries(
  faer_tests
  PUBLIC faer
         CONAN_PKG::fmt
         CONAN_PKG::eigen
)
target_compile_options(faer_tests PUBLIC -fmessage-length=100)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(backward_cpp_main OBJECT src/backward.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)

add_executable(test.matrix.mul src/matrix_mul.cpp)

target_link_libraries(
  test.matrix.mul
  faer_tests
  backward_cpp_main
  doctest_main
)
