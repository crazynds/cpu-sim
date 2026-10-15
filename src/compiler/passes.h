#pragma once

// The IR passes of cpusim-devc around the SIMT pass (passes.cpp; each one's
// reasons are at its definition). main.cpp runs them in this order: the
// lowerings on the per-thread code, then the SIMT pass, then the clean-ups of
// the warp code.

#include <string>

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

namespace cpusim {

// --- the per-thread code, before the SIMT pass
// NVVM atomic increment/decrement and memory barriers as LLVM's.
void lower_nvvm_atomics(llvm::Module& m);
// __device__ and __constant__ variables: the symbol table lines of the manifest.
std::string lower_device_globals(llvm::Module& m);
// __syncthreads_count/_and/_or as a counter in shared memory around a barrier.
void lower_barrier_votes(llvm::Module& m, llvm::Function& kernel);
// Device printf as calls the runtime formats.
void lower_printf(llvm::Function& f);
// Device malloc of 2 GiB or more returns null, as on a GPU.
void cap_device_malloc(llvm::Function& f);
// __hfma: the CPU's half FMA with AVX512-FP16, else its emulation.
void lower_half_fma(llvm::Module& m, bool fp16);
// fptrunc(sqrt(fpext h)) as a half square root.
void shrink_half_sqrt(llvm::Function& f);
// memcpy/memmove/memset as plain loads and stores; true if any changed.
bool expand_mem_intrinsics(llvm::Function& f, llvm::TargetMachine* tm);
// Loops that wait for another thread's write get a yield point; true if any.
bool mark_spin_loops(llvm::Function& f);
// Loops with a barrier are not unrolled.
void keep_barrier_loops_rolled(llvm::Function& f);
bool has_alloca(const llvm::Function& f);

// What a kernel's per-thread code needs (CPUSIM_DEVC_FEATURES=1 prints it):
// the most 32-bit units of varying values live at once (at W lanes they take
// units * W / 16 zmm registers), the instructions, the long-latency ones.
struct Features {
    unsigned live_units = 0, instructions = 0, loads = 0, stores = 0, slow_ops = 0, loops = 0;
};
Features kernel_features(llvm::Function& f);

// --- the warp code, after the SIMT pass
// Masks under a uniform guard become the guarded mask; true if any changed.
bool refine_uniform_masks(llvm::Function& f);
// select on wide i1 vectors as select on integers (an LLVM 19 hang).
void widen_mask_selects(llvm::Module& m);

}  // namespace cpusim
