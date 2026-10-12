# 28. Function calls that are not inlined

> **In one sentence:** a device function that cannot be inlined (a recursive one, or one called through a pointer or a vtable) now gets a W-lane version that takes the warp's lane mask and every argument as a vector; a recursion ends because the block holding the call only runs while some lane is active, and a call through a pointer runs once per distinct target among the active lanes.

The general concepts (masks, divergence, variants of W lanes) are in [00-basic-concepts.md](../00-basic-concepts.md).

## The problem

Until now every device function was inlined into the kernels ([DESIGN.md](../../../DESIGN.md) 5.6): `cpusim-devc` marks them `alwaysinline` and the SIMT pass works on one flat per-thread function. That covers ordinary calls, but not:

- **recursion** (`fib`, a tree traversal, divide and conquer): the inliner cannot expand a function into itself, so the call stayed and the SIMT pass stopped with "call to a function that could not be inlined";
- **calls through a pointer**: a table of `__device__` function pointers indexed per thread, and virtual functions, whose target is read from the object's vtable. The pass rejected every indirect call.

## The idea

A warp executes a call the same way it executes any other instruction: all lanes at once, under the current mask. So the callee needs a version that does for its body what the kernel entry does for the kernel's: run W lanes with masks.

`__cpusim_fn<W>_<name>(ctx, mask, a0, ..., an)` takes the context (for `threadIdx`, `blockIdx` and the rest), the mask of the lanes that make the call, and every argument as a `<W x T>` vector. It returns a `<W x T>`. Its body is built by the same `WarpBuilder` as a kernel, from the same per-thread code, with three differences:

| | Kernel entry | Function version |
|---|---|---|
| Initial mask | lanes inside the block (`ctx` or a constant) | the `mask` argument |
| Arguments | uniform, read from the launch's parameter buffer | varying, one per lane, in registers |
| `ret` | the lanes are done | each lane writes its value to a return slot under its mask; the function returns the slot |

**Termination.** A recursive call sits in some block of the linearized function, and the pass only enters a block when its mask has some lane ([00](../00-basic-concepts.md)). In `fib`, the block with `fib(n - 1) + fib(n - 2)` has the mask of the lanes with `n >= 2`; when no lane has it, the block is skipped and the recursion unwinds. Lanes that finish early stay inactive in the deeper calls, exactly as with a divergent loop.

**Calls through a pointer.** The target can differ per lane. The call becomes a loop: take the target of the first remaining lane, form the mask of every remaining lane that has the same target, call that target's vector version with that mask, and merge the result under it. It repeats while lanes remain. When the target is uniform (the same for the whole warp), there is one iteration and no loop. The candidate targets are the device functions with the same type whose address is taken; the target is found by comparing the pointer with each of them.

**Function addresses.** The per-thread functions are not compiled for x86 (they still have NVPTX intrinsics). Once every call is replaced, they are deleted. Their address, if still used (a table of pointers, a vtable, the comparisons above), becomes the address of a one-byte global of its own. It is enough for what device code does with a function pointer: store it, load it and compare it.

## Step by step

1. **Uniformity** (`src/compiler/uniformity.cpp:46-49`): a new flag makes the arguments of the function varying from the start; the rest of the analysis is unchanged.
2. **What is accepted** (`src/compiler/simt.cpp:301-306`, `is_device_call`): a call to a defined, non-variadic, non-intrinsic function, or any call through a pointer. Inside a function version, `__syncthreads` and direct use of `__shared__` variables are rejected with a clear message (a pointer into shared memory passed as an argument works, since it is a plain address).
3. **Declaration** (`src/compiler/simt.cpp:1706-1727`, `vector_fn`): on the first call, the version is declared (`internal`, `fastcc`, so the vectors travel in registers) and put on a list. Aggregate arguments or results (a `float3` returned by value) are rejected for now.
4. **The call** (`src/compiler/simt.cpp:1744-1811`, `emit_call`): direct calls become one call with the block's mask; calls through a pointer become the loop over distinct targets.
5. **The body** (`src/compiler/simt.cpp:3318-3333`): after a kernel variant is built, each pending version is built from a copy of the per-thread function, which may declare more (the recursion declares itself; it is already there).
6. **Return** (`src/compiler/simt.cpp:2846-2850`): a `ret` stores its value to the return slot, blending under the mask when the block is divergent; the end of the function returns the slot.
7. **Clean up** (`src/compiler/main.cpp:604-622`): the per-thread functions are deleted, and their remaining addresses replaced.

Two problems surfaced on the way, outside the compiler pass:

- **vtables did not link.** The device module is a shared object, and a `linkonce_odr` global with default visibility (a vtable, `_ZTV3Tri`) may be preempted, so the linker refuses PC-relative addressing to it. Non-local globals are now `protected` (`src/compiler/main.cpp:307-312`): still visible to `dlsym`, no longer preemptible.
- **math in `__host__ __device__` functions.** `cuda_runtime.h` did not include `<math.h>`, so a function compiled for both sides found only the `__device__` `sqrtf`. It is included now, as nvcc does.

The runtime also gained `cudaDeviceSetLimit` and `cudaDeviceGetLimit`. They record the values. The stack size does not constrain anything here: device code runs on the worker threads' stacks, which are megabytes, not CUDA's 1 KiB default.

## Example

```cpp
__host__ __device__ unsigned fib(unsigned n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
```

becomes (32-lane version, simplified):

```llvm
define internal fastcc <32 x i32> @__cpusim_fn32__Z3fibj(ptr %ctx, <32 x i1> %mask, <32 x i32> %n) {
  ; lanes with n >= 2: the recursive block
  %a = call fastcc <32 x i32> @__cpusim_fn32__Z3fibj(ptr %ctx, <32 x i1> %rec, <32 x i32> %n1)
  %b = call fastcc <32 x i32> @__cpusim_fn32__Z3fibj(ptr %ctx, <32 x i1> %rec, <32 x i32> %n2)
  ; lanes with n < 2 return n, the others a + b, each under its mask
}
```

## Results

`bench/calls.cu` runs the same `__host__ __device__` functions as kernels (one element per thread) or in an OpenMP loop. The checksums agree among the three. Simulator through the driver on 28 CPUs; OpenMP with 32 threads, run right after; GPU = RTX 5060 Laptop (stack raised to 16 KiB). Millions of elements per second, higher is better:

| Workload | cucc | OpenMP | GPU |
|---|---|---|---|
| fib, n = 20 | 7.0 | 1.2 | 1.8 |
| fib, n per element 18..23 | 1.7 | 0.7 | 0.2 |
| BST search, 4M nodes (recursive) | 640.6 | 180.8 | 1418.2 |
| divide and conquer, 256 leaves | 312.0 | 30.7 | 139.2 |
| function pointers, 4 per element | 2727.5 | 939.5 | 8300.2 |
| virtual calls, grouped by type | 1796.8 | 1735.7 | 17805.8 |
| virtual calls, mixed types | 1720.6 | 1273.1 | 17694.8 |

The cucc column varied between runs by up to 2x in the recursive workloads (a second run gave 0.9 for fib per element and 336 for the BST search), with the laptop's thermal state; the ranking against OpenMP did not change.

Why the simulator wins where it does:

- **Recursion:** one call does the work of 32 lanes; the OpenMP thread makes one scalar call per element. The GPU loses here: deep recursion spills to its local memory, and per-lane depths serialize the warp.
- **BST search:** the 32 lanes' loads of the next node are one gather, so 32 cache misses are in flight per call instead of one.
- **Virtual calls, grouped by type:** almost even. The OpenMP loop also gets a predictable indirect branch, and both are limited by memory (16M objects, 256 MiB).

## Limitations

- Arguments and results must be scalars (integers, floating point, pointers): no structs by value, and no per-thread arrays passed by pointer (they live in a lane-major layout that has no per-lane address).
- No `__syncthreads` or direct `__shared__` access inside a function that is not inlined.
- A function's address is local to the module that took it. Each block shape compiled by the JIT is its own module, so a pointer stored by one kernel and called by a kernel launched with another block shape does not match any target. This is the same problem as `__device__` variables across modules, open in the stress test.
- Every argument is varying, even one that is the same for the whole warp (the tree pointer in the BST search): inside the function, accesses through it are classified at run time instead of at compile time.

Previous: [27](27-kernel-at-peak.md). Back to the [iteration 3 index](README.md).
