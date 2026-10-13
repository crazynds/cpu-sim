#pragma once

#include <string>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace cpusim {

struct ParamSlot {
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct KernelInfo {
    std::string name;            // mangled
    llvm::Function* scalar = nullptr;
    llvm::Function* entry = nullptr;
    llvm::Function* profiled = nullptr;
    bool has_barriers = false;
    bool spins = false;  // waits for other warps in a loop (yield points): 32-lane variants only
    uint32_t spec_block[3] = {0, 0, 0};  // block shape the code is specialized for (0: any)
    uint32_t wide_variants = 0;  // CPUSIM_META_WIDE_SHIFT / CPUSIM_META_SUSP_SHIFT bits
    std::vector<ParamSlot> params;
    uint64_t param_bytes = 0;
    uint64_t shared_bytes = 0;  // static __shared__ (set by lower_shared_memory)
    uint64_t frame_bytes = 0;   // per-warp state saved at barriers (0: no barrier), 32-lane variants
    uint64_t frame_bytes_wide[2] = {};  // suspending variants of 2 and 4 warps
    // Threads per warp (warpSize: 32, or 16 or 64 with CPUSIM_WARP_SIZE), and
    // the lanes of the warp entry: the warp size, or fewer when the whole
    // block is smaller (a block of 16 threads with warps of 32 runs in one
    // 16-lane vector; the missing lanes are threads that do not exist).
    unsigned warp_size = 32;
    unsigned warp_lanes = 32;
    bool traced = false;  // has __cpusim_trace_<name> (wmma GEMM candidate)
};

// Builds `__cpusim_entry_<name>`: the per-thread kernel turned into code that
// runs a whole warp (32 lanes) with masks. On failure returns false and
// describes what is not supported yet in *error.
// Name of the function standing for the block's shared memory base (uniform).
constexpr const char* kSharedBaseFn = "__cpusim_shared_base";
// Warp primitives the device header builds __shfl_* and __ballot_* on.
constexpr const char* kShflFn = "__cpusim_shfl";
constexpr const char* kBallotFn = "__cpusim_ballot";
// `__cpusim_yield()`: in a loop that waits for memory another warp writes
// (cpusim-devc inserts it): the warp suspends there as at a barrier and lets
// the other warps of its block run.
constexpr const char* kYieldFn = "__cpusim_yield";
inline bool is_yield(const llvm::Instruction& i) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(&i);
    return call && call->getCalledFunction() && call->getCalledFunction()->getName() == kYieldFn;
}

// Tensor cores: `__cpusim_mma_<types>(d, a, b, c)`, pointers to the fragments
// (per-thread arrays) of one warp's 16x16x16 multiply-accumulate (mma.h). The
// warp code calls `__cpusim_tensor_<types>` of libcpusim_tensor per warp.
constexpr const char* kMmaPrefix = "__cpusim_mma_";
constexpr const char* kTensorPrefix = "__cpusim_tensor_";
// `float __cpusim_mmav_<types>(int j, a, b0..b7, c0..c7)`: element j of
// D = A * B + C for the calling thread, A by pointer (its fragment), B and C
// by value; the 8 calls of one mma_sync share their operands and become one
// inline mma (emit_mmav). Fragments passed by value can live in registers.
constexpr const char* kMmavPrefix = "__cpusim_mmav_";
inline bool is_mmav(const llvm::Value* v) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(v);
    return call && call->getCalledFunction() && call->getCalledFunction()->getName().starts_with(kMmavPrefix);
}
inline bool is_mma(const llvm::Value* v) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(v);
    return call && call->getCalledFunction() && call->getCalledFunction()->getName().starts_with(kMmaPrefix);
}

// Calls `T __cpusim_cond_load.<T>(ptr p, i1 c, T other)`: "c ? *p : other",
// reading memory only when c holds. Introduced by if-conversion.
constexpr const char* kCondLoadPrefix = "__cpusim_cond_load.";
inline bool is_cond_load(const llvm::Value* v) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(v);
    return call && call->getCalledFunction() && call->getCalledFunction()->getName().starts_with(kCondLoadPrefix);
}

// A matrix multiplication written with wmma: builds __cpusim_trace_<name>,
// the kernel's scalar copy that records its wmma operations instead of doing
// them (wmma_trace.cpp). False, with the reason, if the kernel cannot be one.
bool build_wmma_trace(llvm::Module& m, KernelInfo& kernel, std::string* why);

// Gives every __shared__ variable used by the kernel an offset in the block's
// shared memory and rewrites its uses as shared base + offset.
void lower_shared_memory(llvm::Module& m, KernelInfo& kernel);

// Builds `__cpusim_entry_<name>` (lanes = 32, one warp, barriers suspend the
// warp), or with lanes > 32 the block-wide `__cpusim_entry<lanes>_<name>`
// (used only when the whole block fits, so barriers are no-ops). With
// `profile`, builds the instrumented 32-lane `__cpusim_prof_<name>` instead.
// The scalar function is modified, so build each variant from its own copy.
// `suspending` (only meaningful with lanes > 32): barriers save state and
// return, as in the 32-lane entry, instead of being no-ops.
// `cooperative`: half of a block per call (the SMT sibling runs the other
// half), barriers call ctx->coop_barrier: `__cpusim_entry<lanes>c_<name>`.
bool build_warp_entry(llvm::Module& m, KernelInfo& kernel, llvm::Function& scalar, bool profile, unsigned lanes,
                      bool suspending, std::string* error, bool cooperative = false);

}  // namespace cpusim
