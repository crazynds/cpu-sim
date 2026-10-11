#pragma once

#include <string>
#include <vector>

namespace cpusim {

// A physical core and its SMT siblings (logical CPU numbers, ascending).
struct PhysicalCore {
    int package = 0;
    int core_id = 0;
    std::vector<int> cpus;
};

struct Topology {
    std::vector<PhysicalCore> cores;   // ordered by first logical CPU
    std::string cpu_model;             // "AMD Ryzen 9 8940HX with Radeon Graphics"
    int logical_cpus = 0;
};

// Reads /sys/devices/system/cpu; never assumes a numbering scheme for siblings.
Topology read_topology();

// The CPU model without marketing suffixes ("AMD Ryzen 9 8940HX").
std::string short_cpu_model(const std::string& model);

// True if the CPU (and OS) support the AVX-512 subset the generated code needs.
bool cpu_has_avx512();

// What the tensor core emulation (mma.h) can run natively, the CPU's answer
// to a GPU's tensor core generation. Each level includes the ones below:
//   Fp16: half inputs through F16C and FMA (every AVX-512 CPU);  sm_70
//   Int8: + int8 dot products (AVX512-VNNI vpdpbusd);           sm_75
//   Bf16: + bfloat16 dot products (AVX512-BF16 vdpbf16ps);       sm_80 and up
// CPUSIM_TENSOR=fp16|int8|bf16 lowers it (never above what the CPU has), to
// see how a program behaves on a smaller CPU.
enum class TensorLevel { Fp16 = 0, Int8 = 1, Bf16 = 2 };
TensorLevel tensor_level();
const char* tensor_level_isa(TensorLevel level);  // "AVX512-BF16 + AVX512-VNNI"
// The compute capability reported for a level (major * 10 + minor).
int compute_capability(TensorLevel level);

}  // namespace cpusim
