target_compile_options(faer PUBLIC "-march=native")

# Conan dependencies
set(CONAN_REQUIRES
    fmt/7.1.2
    eigen/3.3.9
    blaze/3.8
    backward-cpp/1.5
    benchmark/1.5.2
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

add_executable(bench.matrix.mul.large src/matrix_mul_large.cpp)
target_link_libraries(bench.matrix.mul.large faer_bench bench_backward_cpp_main)

add_executable(bench.matrix.mul.small.cc src/matrix_mul_small_cc.cpp)
target_link_libraries(
  bench.matrix.mul.small.cc faer_bench bench_backward_cpp_main
)

add_executable(bench.matrix.mul.small.cr src/matrix_mul_small_cr.cpp)
target_link_libraries(
  bench.matrix.mul.small.cr faer_bench bench_backward_cpp_main
)

add_executable(bench.matrix.mul.small.rc src/matrix_mul_small_rc.cpp)
target_link_libraries(
  bench.matrix.mul.small.rc faer_bench bench_backward_cpp_main
)

add_executable(bench.matrix.mul.small.rr src/matrix_mul_small_rr.cpp)
target_link_libraries(
  bench.matrix.mul.small.rr faer_bench bench_backward_cpp_main
)
