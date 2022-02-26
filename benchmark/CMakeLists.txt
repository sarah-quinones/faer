target_compile_options(faer PUBLIC "-march=native")

# Conan dependencies
set(CONAN_REQUIRES
    fmt/7.1.2
    eigen/3.3.9
    blaze/3.8
    backward-cpp/1.5
    benchmark/1.6.0
)
set(CONAN_OPTIONS)
run_conan()

add_library(bench_backward_cpp_main OBJECT ../test/src/backward.cpp)
add_library(faer_bench src/null.cpp)

target_link_libraries(bench_backward_cpp_main CONAN_PKG::backward-cpp)
target_link_libraries(
  faer_bench
  PUBLIC faer
         CONAN_PKG::benchmark
         CONAN_PKG::fmt
         CONAN_PKG::eigen
         CONAN_PKG::blaze
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(faer_bench PUBLIC -ftime-trace)
endif()

add_executable(bench.matmul.large src/matmul_large.cpp)
target_link_libraries(bench.matmul.large faer_bench bench_backward_cpp_main)
