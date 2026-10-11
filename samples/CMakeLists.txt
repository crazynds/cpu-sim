cmake_minimum_required(VERSION 3.24)
project(CudaSimulatorSamples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# No fused multiply-add on the GPU, so float results match a non-FMA CPU
# reference bit for bit wherever the math is otherwise IEEE exact.
option(SAMPLES_NO_FMAD "Compile device code with -fmad=false" ON)

set(SAMPLES
    matmul
    matmul_naive
    nbody
    reduction
    histogram
    heat2d
    scan
    transpose
    bitonic_sort
    spmv
    mandelbrot
    tensor_gemm
    cpu_gemm
)

enable_testing()

foreach(sample IN LISTS SAMPLES)
    add_executable(${sample} ${sample}.cu)
    target_include_directories(${sample} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
    if(SAMPLES_NO_FMAD)
        target_compile_options(${sample} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
    endif()
    add_test(NAME ${sample} COMMAND ${sample})
endforeach()
