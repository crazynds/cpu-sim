cmake_minimum_required(VERSION 3.24)
project(CudaSimulator VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

# Shared by the driver, both device backends and the tools.
add_library(cpusim_common STATIC
    src/common/config.cpp
    src/common/error.cpp
    src/common/paths.cpp
    src/common/sensors.cpp
    src/common/topology.cpp
    src/common/units.cpp
)
target_include_directories(cpusim_common PUBLIC src)
target_compile_definitions(cpusim_common PUBLIC CPUSIM_VERSION="${PROJECT_VERSION}")
target_link_libraries(cpusim_common PUBLIC Threads::Threads)

# Device interface: VRAM, allocator and the two backends (in-process and daemon client).
add_library(cpusim_device STATIC
    src/device/allocator.cpp
    src/device/daemon_client.cpp
    src/device/device.cpp
    src/device/debug_exec.cpp
    src/device/device_core.cpp
    src/device/executor.cpp
    src/device/inprocess.cpp
    src/device/inprocess_status.cpp
    src/device/properties.cpp
    src/device/vram.cpp
)
target_link_libraries(cpusim_device PUBLIC cpusim_common ${CMAKE_DL_LIBS})

# Tensor core emulation (mma_sync), linked by the kernel modules that use it.
# Built for baseline x86-64: each routine enables its own instruction set and
# only runs on CPUs that have it (the runtime checks before a launch).
add_library(cpusim_tensor SHARED src/tensor/mma.cpp)
target_compile_options(cpusim_tensor PRIVATE -O3)

# Our CUDA runtime replacement, linked into programs built by cucc.
# Named libcpusim_cudart so it never collides with NVIDIA's libcudart on the
# library path (LD_LIBRARY_PATH would win over our RUNPATH).
add_library(cudart SHARED src/cudart/runtime.cpp src/cudart/streams.cpp)
set_target_properties(cudart PROPERTIES OUTPUT_NAME cpusim_cudart)
target_include_directories(cudart PUBLIC include)
target_link_libraries(cudart PRIVATE cpusim_device)
add_dependencies(cudart cpusim_tensor)  # kernel modules link it

add_executable(cpusimd
    src/driver/main.cpp
    src/driver/server.cpp
    src/driver/monitor.cpp
)
target_link_libraries(cpusimd PRIVATE cpusim_device)

add_executable(cpu-smi src/smi/main.cpp)
target_link_libraries(cpu-smi PRIVATE cpusim_device)

add_executable(device_query tools/device_query.cpp)
target_link_libraries(device_query PRIVATE cudart)

add_executable(memtest tools/memtest.cpp)
target_link_libraries(memtest PRIVATE cudart)

# Device compiler (needs LLVM 19). Without root, the Ubuntu packages can be
# unpacked with `apt-get download` + `dpkg -x` (see README).
set(CPUSIM_LLVM_ROOT "$ENV{HOME}/.local/opt/llvm19/usr/lib/llvm-19" CACHE PATH "LLVM 19 installation")
find_package(LLVM CONFIG HINTS "${CPUSIM_LLVM_ROOT}/lib/cmake/llvm" /usr/lib/llvm-19/lib/cmake/llvm)
if(LLVM_FOUND AND LLVM_VERSION_MAJOR EQUAL 19)
    message(STATUS "LLVM ${LLVM_PACKAGE_VERSION}: ${LLVM_INSTALL_PREFIX}")
    add_executable(cpusim-devc
        src/compiler/main.cpp
        src/compiler/simt.cpp
        src/compiler/uniformity.cpp
    )
    target_include_directories(cpusim-devc PRIVATE src SYSTEM ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(cpusim-devc PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(cpusim-devc PRIVATE -fno-rtti -Wno-unused-parameter)
    target_link_libraries(cpusim-devc PRIVATE LLVM)
    set_target_properties(cpusim-devc PROPERTIES BUILD_RPATH "${LLVM_LIBRARY_DIR}")
    set(CPUSIM_CLANG "${LLVM_TOOLS_BINARY_DIR}/clang++" CACHE FILEPATH "clang++ used by cucc")

    # cucc: the nvcc replacement, configured with the paths of this build.
    set(CPUSIM_DEVC "${CMAKE_BINARY_DIR}/cpusim-devc")
    set(CPUSIM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
    set(CPUSIM_CUDA_STUB "${CMAKE_SOURCE_DIR}/share/cuda-stub")
    set(CPUSIM_LIB_DIR "${CMAKE_BINARY_DIR}")
    # The runtime recompiles device bitcode at launch (JIT) with these tools.
    target_compile_definitions(cudart PRIVATE CPUSIM_DEVC_PATH="${CPUSIM_DEVC}" CPUSIM_CLANG_PATH="${CPUSIM_CLANG}"
                                              CPUSIM_LIB_DIR_PATH="${CPUSIM_LIB_DIR}")
    configure_file(tools/cucc.in ${CMAKE_BINARY_DIR}/cucc @ONLY
                   FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
else()
    message(WARNING "LLVM 19 not found: the device compiler (cpusim-devc, cucc) will not be built")
endif()

enable_testing()

add_executable(test_allocator tests/test_allocator.cpp)
target_link_libraries(test_allocator PRIVATE cpusim_device)
add_test(NAME allocator COMMAND test_allocator)

add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config PRIVATE cpusim_common)
add_test(NAME config COMMAND test_config)

add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE cudart)
add_test(NAME runtime_inprocess COMMAND test_runtime)
set_tests_properties(runtime_inprocess PROPERTIES ENVIRONMENT "CPUSIM_BACKEND=inprocess;CPUSIM_VRAM=1GiB")

# Starts a private daemon, runs the same runtime test through it and checks cpu-smi.
add_test(NAME runtime_daemon
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/daemon_test.sh
            $<TARGET_FILE:cpusimd> $<TARGET_FILE:test_runtime> $<TARGET_FILE:cpu-smi>)

# Self-checking tests of single features (tests/cuda), each run with every
# lane width, without JIT and on the daemon backend.
if(TARGET cpusim-devc)
    foreach(feature IN ITEMS warp_ops math division local_arrays streams async managed types wmma streaming)
        add_test(NAME cuda_${feature}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_feature_test.sh ${CMAKE_BINARY_DIR}/cucc
                    $<TARGET_FILE:cpusimd> ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/${feature}.cu)
        set_tests_properties(cuda_${feature} PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL" LABELS features TIMEOUT 120)
    endforeach()
    # Tensor cores on CPUs with fewer dot product instructions: lower compute
    # capability, and the kernels that need more must not launch.
    set_tests_properties(cuda_wmma PROPERTIES ENVIRONMENT "FEATURE_ENVS=CPUSIM_TENSOR=int8 CPUSIM_TENSOR=fp16")
    # Compares durations (concurrent streams): alone, or other tests' load
    # decides the result.
    set_tests_properties(cuda_async PROPERTIES RUN_SERIAL TRUE)

    # Stability: launch geometry, divergent control flow and repeated launches
    # under the width tuning, in every mode as above; then bitwise
    # reproducibility of the results across all modes and runs, and of the
    # compiler output.
    foreach(feature IN ITEMS launch_shapes control_flow repeat_launch)
        add_test(NAME stability_${feature}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_feature_test.sh ${CMAKE_BINARY_DIR}/cucc
                    $<TARGET_FILE:cpusimd> ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/${feature}.cu)
        set_tests_properties(stability_${feature} PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL" LABELS stability TIMEOUT 120)
    endforeach()
    add_test(NAME stability_repro
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_repro_test.sh ${CMAKE_BINARY_DIR}/cucc
                $<TARGET_FILE:cpusimd> ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/repro.cu)
    set_tests_properties(stability_repro PROPERTIES PASS_REGULAR_EXPRESSION "repro reproducibility: PASS"
                                                    FAIL_REGULAR_EXPRESSION "FAIL" LABELS stability TIMEOUT 120)
endif()

# CUDA samples compiled by our own toolchain (the proof of concept targets).
if(TARGET cpusim-devc)
    # The samples carry GPU-sized workloads (>= 100 ms on an RTX 5060 where the
    # kernel allows it); the tests build them with -DSAMPLE_TEST (smaller sizes,
    # same code paths) and OpenMP for the CPU reference: seconds each.
    foreach(sample IN ITEMS matmul_naive matmul transpose heat2d scan bitonic_sort mandelbrot tensor_gemm cpu_gemm)
        add_test(NAME sample_${sample}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_sample_test.sh ${CMAKE_BINARY_DIR}/cucc
                    $<TARGET_FILE:cpusimd> ${CMAKE_CURRENT_SOURCE_DIR}/samples/${sample}.cu -fmad=false -fopenmp -DSAMPLE_TEST)
        set_tests_properties(sample_${sample} PROPERTIES PASS_REGULAR_EXPRESSION "${sample}: PASS.*${sample}: PASS"
                                                         FAIL_REGULAR_EXPRESSION "FAIL" LABELS samples TIMEOUT 120)
    endforeach()
endif()
