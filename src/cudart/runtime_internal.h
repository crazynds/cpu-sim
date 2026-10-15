#pragma once

// Internal to the CUDA runtime (src/cudart): the state its parts share.
// runtime.cpp: the device, streams, events, errors and settings; module.cpp:
// the modules the compiler registers, their JIT and symbols; launch.cpp:
// kernel launches, attributes and profiling; memory.cpp: allocations and
// copies; graph.cpp: CUDA graphs and stream capture.

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cudart/streams.h"
#include "cuda_runtime_api.h"
#include "device/device.h"
#include "device/kernel_abi.h"

// Not part of the library's interface: hidden from the programs it is linked to.
#pragma GCC visibility push(hidden)

namespace cpusim::rt {

using cpusim::Error;

struct Runtime {
    std::unique_ptr<cpusim::Device> device;
    Error init_error = Error::Success;
    // After `device`: destroyed first, so the queued work still has a device.
    cpusim::cudart::Scheduler sched;
};

Runtime& runtime();
extern thread_local cudaError_t last_error;
// Threads per warp the program's kernels were compiled for (CPUSIM_WARP_SIZE
// at compile time, in each module's manifest): what warpSize reports.
extern std::atomic<int> program_warp_size;
int warp_size();
// Records a failure as the last error, and returns it.
cudaError_t record(Error e);
cudaError_t record(cudaError_t err);
// At a synchronization point: the first error of asynchronous work, if any.
cudaError_t async_error();
// Returns the device or records why there is none.
cpusim::Device* device(cudaError_t* err);

#define REQUIRE_DEVICE(dev)                    \
    cudaError_t init_err_ = cudaSuccess;       \
    cpusim::Device* dev = device(&init_err_);  \
    if (!dev) return init_err_

// ---- modules registered by the compiler-generated constructors (module.cpp)

struct KernelDesc {
    std::string name;
    uint64_t param_bytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> params;  // offset, size
    int min_sm = 0;  // compute capability (major * 10 + minor) it needs, 0: any
    uint64_t shared_bytes = 0;  // static __shared__
    int max_dynamic_shared = -1;  // cudaFuncAttributeMaxDynamicSharedMemorySize (-1: the default)
    bool lane_independent = false;  // its lanes never look at each other (CPUSIM_META_LANE_INDEPENDENT)
    bool barriers = false;          // waits at barriers: specialized for exact shapes, never for classes
};

// A module recompiled for one block shape (JIT at launch).
struct JitModule {
    bool tried = false, ok = false;
    uint64_t handle = 0;
    std::map<std::string, uint64_t> functions;
};

// A __device__ or __constant__ variable: one copy in the VRAM for the whole
// program, allocated when the generic module is loaded.
struct SymbolDesc {
    std::string name;
    uint64_t size = 0, align = 1;
    uint64_t address = 0;
    bool managed = false;  // __managed__: kept equal to `host` around launches
    void* host = nullptr;  // its host copy (the registered shadow)
    std::vector<uint8_t> initial;  // managed: its initial value, for the host copy
};

struct Module {
    const cpusim_fatbin_header* header = nullptr;
    std::map<std::string, KernelDesc> kernels;
    std::vector<SymbolDesc> symbols;
    // For Device::bind_symbols: "<address> <size> <name>" per line.
    std::string symbol_table() const {
        std::string t;
        for (const auto& s : symbols) t += std::to_string(s.address) + " " + std::to_string(s.size) + " " + s.name + "\n";
        return t;
    }
    bool loaded = false;
    uint64_t handle = 0;
    // Its code was compiled for another kind of CPU (common/cpu_id.h): its
    // kernels do not launch (cudaErrorNoKernelImageForDevice), as a CUDA
    // binary without code for the GPU it runs on.
    bool foreign = false;
    std::map<std::array<uint32_t, 3>, JitModule> jit;  // by block shape
    // Compiled at run time, one kernel for one shape (the ahead-of-time
    // modules hold only the kernels launched with their constant shape).
    std::map<std::pair<std::array<uint32_t, 3>, std::string>, JitModule> jit_kernel;
    std::map<std::array<uint32_t, 3>, bool> dumped;    // CPUSIM_DUMP_ASM, by block shape ({0,0,0}: generic)
};

struct Function {
    Module* module = nullptr;
    const KernelDesc* kernel = nullptr;
    uint64_t handle = 0;  // device function, resolved on first launch
};

struct Variable {
    Module* module = nullptr;
    std::string name;  // device name
    size_t size = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<std::unique_ptr<Module>> modules;
    std::map<const void*, Function> functions;  // host stub -> kernel
    std::map<const void*, Variable> variables;  // host shadow -> __device__ / __constant__ variable
};

Registry& registry();
Error load_generic(Module& m, cpusim::Device* d);
// The device function of kernel `k` for blocks of `shape`: the module
// compiled for that shape or class (JIT), else the generic one.
uint64_t jit_function(Module& m, const KernelDesc& k, const std::array<uint32_t, 3>& shape, cpusim::Device* d);
// CPUSIM_DUMP_ASM=1: the x86 assembly of the kernels a program launches.
bool dump_asm_enabled();
void dump_asm(Module& m, const std::array<uint32_t, 3>& block);

// ---- copies (memory.cpp)

cpusim::cudart::CompletionPtr queue_copy(cpusim::Device* d, void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                         cudaStream_t stream, bool sync);
cpusim::cudart::CompletionPtr queue_memset(cpusim::Device* d, void* ptr, int value, size_t count, cudaStream_t stream);
// Host memory registered or allocated as mapped: its device-side shadow.
struct MappedRange {
    char* host;
    char* shadow;  // == host in-process
    size_t size;
};
extern std::mutex mapped_mutex;
extern std::vector<MappedRange> mapped_ranges;

// The legacy and per-thread default streams are not objects.
inline bool is_object(cudaStream_t s) { return s && s != cudaStreamLegacy && s != cudaStreamPerThread; }

}  // namespace cpusim::rt

// ---- events and graphs (graph.cpp)

// An event's last record: the operation that stamps it.
struct CUevent_st {
    unsigned flags = 0;
    std::mutex mutex;
    cpusim::cudart::CompletionPtr pending;  // null: never recorded
    std::shared_ptr<std::chrono::steady_clock::time_point> when;
    CUgraph_st* captured = nullptr;  // recorded in a stream being captured into this graph
};

// ---- CUDA Graphs
//
// A node is an operation with its arguments copied (a kernel's parameter
// values, as cudaLaunchKernel copies them). Nodes are kept in the order they
// were added: each one after the nodes it depends on (in a capture, the
// program issued them in that order; explicit nodes name earlier ones). A
// launch enqueues them in that order on its stream.
struct CUgraphNode_st {
    enum Kind { Kernel, Memcpy, Memset, Host, Empty } kind = Empty;
    const void* func = nullptr;
    dim3 grid, block;
    size_t shared = 0;
    std::vector<uint8_t> args;                         // parameter values, packed
    std::vector<std::pair<size_t, size_t>> arg_slots;  // offset, size in `args`
    void* dst = nullptr;
    const void* src = nullptr;
    size_t count = 0;
    cudaMemcpyKind copy_kind = cudaMemcpyDefault;
    cudaMemsetParams memset{};
    std::vector<uint8_t> pattern;  // one row of a memset with elements wider than a byte
    cudaHostFn_t fn = nullptr;
    void* data = nullptr;
};

struct CUgraph_st {
    std::vector<std::unique_ptr<CUgraphNode_st>> nodes;
};

struct CUgraphExec_st {
    std::vector<CUgraphNode_st> nodes;
    std::map<const CUgraphNode_st*, size_t> index;  // the graph's node -> ours
};

namespace cpusim::rt {

// The streams being captured, and into which graph.
extern std::mutex capture_mutex;
extern std::map<cudaStream_t, CUgraph_st*> capturing_streams;
// The graph `stream` is being captured into, or null.
CUgraph_st* capturing(cudaStream_t stream);
CUgraphNode_st* add_node(CUgraph_st* g, CUgraphNode_st::Kind kind);
// A kernel node's parameters copied from `args`.
cudaError_t set_kernel(CUgraphNode_st* n, const void* func, dim3 grid, dim3 block, size_t shared, void** args);
void set_memset(CUgraphNode_st* n, const cudaMemsetParams& p);

}  // namespace cpusim::rt

#pragma GCC visibility pop
