# CudaSimulator: design

Design document consolidated from the planning phase (2026-10-08). It serves as a starting point for whoever continues the implementation: what has already been decided, why, and in what order to build.

There is not yet a name for the project or for the tools. Names used here (`cpusim`, `cpu-smi`) are provisional.

---

## 1. Goal

Take a CUDA program that works on an NVIDIA GPU and, **without changing the source code**, compile it to run on the CPU, with each warp executed on AVX-512 register lanes.

- **AOT compilation**, not interpretation. The kernel written "per thread" becomes native code that executes a whole warp in SIMD. Scheduling, barriers and block control stay in the generated code plus a runtime of our own.
- **Resident driver**, which behaves like the NVIDIA driver: it reserves cores and memory ("VRAM"), and receives kernel modules, parameters and launches.
- **Nothing from NVIDIA is used or distributed**: CUDA headers, runtime and driver are ours. The simulated CUDA version is **13**.
- **AVX-512 is mandatory.** No fallback for now, but the code must detect its absence and emit a clear message.

First milestone: a **basic proof of concept that works end to end** (section 9).

---

## 2. Current state

Updated on 2026-10-08 (see also `README.md`):

- **Base driver done** (step 1 of section 8): `cpusimd`, `cpu-smi`, daemon and in-process backends, VRAM with allocator and per-process contexts, memcpy via `process_vm_readv/writev`, modules via memfd + `dlopen`, launch.
- **Proof of concept reached**: `samples/matmul_naive.cu` (`matmul` with only the naive kernel), unmodified, compiled by `cucc` and executed on both backends prints `PASS`, with a bit-for-bit error equal to that of the RTX 5060 (3.24e-05 with `-fmad=false`). Time: 2.0 ms against the GPU's 0.244 ms (118 GFLOP/s on 28 threads).
- **Profiling** (user request): the compiler generates a second version of each kernel that counts operations per basic block; `CPUSIM_PROFILE=1` enables it (section 6.7).
- How the compiler is today (it differs slightly from section 5, which remains the target):
  - Clang generates per-thread IR for NVPTX (`-O2 -Xclang -disable-llvm-passes`), and `cpusim-devc` switches the target to x86-64;
  - the uniformity analysis is ours (sources: `threadIdx`, lane id, allocas, loads through a varying pointer; joins of divergent branches; divergent loop exits);
  - values that cross blocks are demoted to slots, the CFG is linearized with one mask per block (blocks with an empty mask are skipped) and a write to a slot under a partial mask becomes a blend;
  - loads/stores through a varying pointer: address decomposed into uniform base + varying offset, and a test per half-warp chooses between a masked contiguous load, a broadcast or a gather. The test depends only on the offset, so it leaves the loops;
  - a kernel with something not yet supported (`__syncthreads`, atomics, local arrays...) is a compile error.
- **Barriers and shared memory** (2026-10-08): each `__syncthreads` becomes its own block; the warp saves slots and masks in its frame (`ctx->frame`) and returns `BARRIER`; on return a dispatcher restores them and jumps to after the barrier. The executor runs all the warps of the block up to the barrier, in rounds. `__shared__` becomes a uniform base (`ctx->shared`) + offset; the static size and the frame size go in `__cpusim_meta_<kernel>`. The 6 POC samples pass on both backends.
- **Debug mode** (`CPUSIM_DEBUG`, per launch, requested by the program): step by step with guard pages; it is what found the first barrier bug (a frame smaller than the one used by one of the variants).
- Samples still without support: none since 2026-10-11 (`nbody`, `reduction`, `spmv` and `histogram` are tests now).
- **Stress suite** (2026-10-11): 58 programs, 631 cases, all passing on the GPU, run through `cucc` (`tests/stress/`). It took the simulator from 174 passing cases to 629: atomics, device variables, recursion and function pointers, printf/malloc/assert, cooperative groups, libcu++ atomics, graphs, managed memory, ~40 API functions and ~120 math functions were added, and a kernel's fault or hang no longer takes the driver down (docs/optimization/iteration-3/29-stress-suite.md).
- **Optimization (night of 2026-10-08)**, measured with `bench/` (kernels only, hash compared with the GPU):
  - JIT at launch: the fatbin carries the per-thread bitcode; at the first launch of each `blockDim` the runtime recompiles specialized (threadIdx constant or constant + uniform), with a disk cache. It was the largest gain.
  - Width variants (32; 64/128 without a barrier; whole block 64..512 and "suspended" 64s/128s with a barrier) chosen at run time by measurement; block order in bands (swizzle) for cache reuse.
  - Compiler: SIMT if-conversion with guarded loads, uniform loops with real control flow (including with a barrier), rematerialization of invariants, per-thread liveness at barriers, per-warp invariant cache in the frame, access-pattern tests in integer registers and per whole access, address decomposition into uniform base + varying offset (including `or disjoint`, `mul`/`shl` by a uniform).
  - Result (RTX 5060 / CPU): matmul naive 163 / 224 ms; tiled 126 / 304 ms; mandelbrot 95 / 715 ms; transpose 15 and 11 / 142 and 135 ms; heat_step 1.8 / 15.6 ms; scan_blocks 20.6 / 399 ms; bitonic 26 / 709 ms.
  - The laptop varies its clock with temperature: compare variants only in interleaved pairs in the same process.
- Sample workloads increased (user request) so that the kernels last at least 100 ms on the RTX 5060 when the kernel is compute-bound; the memory-bound ones do not reach 100 ms within 8 GB and use the largest reasonable size.

### Development machine environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 8940HX (Zen 4), 16 cores / 32 threads, 2 CCDs |
| AVX-512 | F, DQ, BW, VL, CD, VNNI, BF16, IFMA, VBMI, VBMI2, BITALG, VPOPCNTDQ. **No FP16.** 256-bit units ("double-pumped") |
| Caches | L1d 32 KB/core, L2 1 MB/core, L3 2 x 32 MB |
| Reference GPU | NVIDIA GeForce RTX 5060 Laptop (compute capability 12.0) |
| CUDA SDK | 13.0 at `/usr/local/cuda-13.0` (used **only** to run the samples on the real GPU) |
| Toolchain | GCC 13.3, CMake 3.28. **Clang and LLVM not yet installed** (need `apt install clang llvm-dev`, with sudo) |

---

## 3. Overview

```text
                 program.cu (unmodified)
                          │
                ┌─────────┴──────────┐
                │ our compiler       │  Clang (only as a C++ parser)
                │                    │  + our CUDA 13 headers
                └─────────┬──────────┘
          host code       │        device code (per-thread IR)
              │           │              │
              │           │     our LLVM passes: SIMT vectorization,
              │           │     masks, barriers, memory
              │           │              │
              ▼           │              ▼
      app binary     ◄────┴──── "fatbin": native .so + per-thread bitcode
              │
              │ links with our libcudart
              ▼
     ┌──────────────────────── device interface ──────────────────────────┐
     │  "in-process" backend           │   "daemon" backend (driver)      │
     │  local worker pool              │   socket + driver memory         │
     └─────────────────────────────────┴──────────────────────────────────┘
                                         │
                                  cpu-smi (future)
```

---

## 4. Execution model

### 4.1 Warp

- **Warp = 32 threads semantically** (`warpSize == 32`). A lot of real code assumes this.
- Physically: **2 zmm registers per value** (2 x 16 lanes of 32 bits) and a `__mmask32` mask. The compiler generates IR with `<32 x T>` / `<32 x i1>` and the LLVM backend legalizes it.
- The warp advances in lockstep. There is no *independent thread scheduling* (Volta+): code that depends on lanes of the same warp making progress independently (a spinlock among lanes) is not supported. Documented limitation.

### 4.2 Warp pairs (K) and SMT

- Each hardware thread executes **a pair of interleaved warps (K=2)** to fill the pipeline (on Zen 4, ~1.5 warps of independent operations saturate the FP units). K=2 also leaves the path open to emulate AMD's wave64 (same physical layout, different semantics in the warp intrinsics).
- **In the POC use K=1.** K=2 comes in as an optimization soon after.
- The idea of splitting the 32 zmm into 2 software banks with a "fast switch" was discarded: SMT does that in hardware, with the full register file for each thread.

### 4.3 Blocks and cores

- **One block per physical core.** The core's **two SMT threads** execute warp pairs **of the same block**.
- Shared memory and the warps' frames live in a block-local buffer (hot in the core's L1/L2).
- Distribution of blocks among cores: a global atomic `blockIdx` counter (one `fetch_add` per block).
- Scheduling within the block:
  ```text
  pair = next_pair[phase].fetch_add(1)
  if pair < num_pairs: execute the pair up to a barrier or the end
  else: wait at the 2-participant barrier
  ```
  The counter only grows, so there is no "in execution" state. The reset between phases uses *sense reversal* or counters alternated per phase.
- **Whoever waits at the barrier must not spin freely**: it steals cycles from the sibling thread, which is finishing the last pair. Use `pause` at a minimum; the ideal is `MONITORX/MWAITX` (user mode on AMD).
- **Small blocks** (fewer than 2 pairs): one block per HW thread instead of one per core.
- Pin the threads to SMT siblings by reading `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` (do not assume numbering).
- No CCD-aware scheduling (decided).

### 4.4 Barriers (`__syncthreads`)

- The kernel becomes a **state machine**: each stretch between barriers is a state. A barrier inside a loop works naturally (the loop returns to an earlier state).
- At the barrier the pair **saves only the live values** in a per-warp frame (in the block's buffer) and returns control to the scheduler.
- Warps that finished (`return`) are marked `done`, do not count toward the barrier (CUDA semantics) and are no longer scheduled. A pair with a `done` warp executes with its mask zeroed.
- The two warps of a pair are always in the same state (they leave and arrive at the same barriers).

### 4.5 Divergence

- Divergent `if`: executes both sides with a mask (linearization). **Uniform-branch shortcut is mandatory**: if one side's mask is all zero, skip the whole block.
- Divergent loops: run while there is an active lane; the mask loses lanes as they exit.
- Effective mask of each instruction = the warp's active mask, combined with predicates.
- Default write semantics: **merge masking** (an inactive lane preserves its value). Zero masking only where it makes sense.

### 4.6 Future (outside the POC)

- **Memory yield**: in chains of dependent loads (pointer chasing), emit a prefetch and hand the turn to another pair. Same state machine as the barriers. Off by default; only worth it where we measure a gain.
- Coarsening/interleaving beyond K=2, if measurements show a gain.

---

## 5. Compiler

### 5.1 Front end

- **Clang as a C++ parser**, in CUDA mode, with `-nocudainc -nocudalib`. A `.cu` is full C++ (templates, `constexpr`, the standard library on the host): a parser of our own is out of the question.
- **Our own CUDA headers** (`cuda_runtime.h`, `vector_types.h`, math functions, warp intrinsics, `CUDART_VERSION 13000`). The simulated CUDA version is the one these headers declare; Clang's support for the NVIDIA SDK ceases to matter.
- Host: `<<<>>>` becomes launch calls that our `libcudart` implements; kernel registration through the mechanism Clang emits.
- Device: LLVM IR **per thread**, with `threadIdx` etc. as intrinsics. We retarget to x86-64 (global/shared/constant/local address spaces).

### 5.2 Pass pipeline (ours)

1. **Value classification** (section 5.3)
2. **CFG linearization** with masks + uniform-branch shortcut
3. **Widening** to `<32 x T>` (warp)
4. **Memory**: chooses contiguous, strided or gather/scatter load/store
5. **State machine** at barriers (frames with live values)
6. (later) K=2 interleaving, memory yield
7. Standard O3 + `-mcpu=znver4` codegen

Do not depend on LLVM's auto-vectorizer: it does not vectorize a whole kernel function. The generated IR is explicit (`llvm.masked.load/store/gather/scatter`, `llvm.x86.avx512.*` intrinsics where exact control is needed). Register allocation and scheduling are left to LLVM. Use `llvm-mca` and assembly inspection to check.

References for the same kind of transformation: ISPC (SPMD on SIMD), Region Vectorizer / Whole-Function Vectorization (Moll, Hack, Karrenberg).

### 5.3 Value classification

| Class | Example | Where it lives |
|---|---|---|
| Constant | `2.0f`, `TILE` | immediate |
| Uniform | `blockIdx.x`, parameter pointers, uniform loop counter | scalar register (GPR) or broadcast |
| Affine | `blockIdx.x*blockDim.x + threadIdx.x` | `uniform base + stride * lane` |
| Varying | `indices[i]`, `x * y` with varying `x` | zmm |

The affine class decides the memory access: stride 1 = masked contiguous load/store; constant stride = load + permute or gather; varying = gather/scatter. "Uniform" exists at levels (grid, block, warp); for code generation the warp level matters.

### 5.4 Memory

- AVX-512 gather/scatter use int32 indices + a scalar base. A 64-bit per-lane pointer would require `i64gather` (8 lanes per instruction, 4 per warp): **recognizing "uniform base + int32 offset"** is a mandatory optimization.
- Masked gather/scatter do not access inactive lanes (no fault).
- Conflict in a scatter (two lanes at the same address): AVX-512 writes in increasing lane order (the highest wins). In CUDA it is undefined, so any order is valid.
- Gather is slow on Zen 4 and scatter even more so. Contiguous whenever possible.
- There is no byte gather: loads of varying `uint8_t` become a dword gather + extraction.

### 5.5 Per-thread local arrays

Private arrays of each thread (`float acc[4]`) live on the warp's stack in **SoA layout** (`acc[i][lane]`), like NVIDIA's *local memory* (interleaved per thread). Uniform index = contiguous load; varying index = gather within that layout. It is an internal decision, invisible to the user.

### 5.6 Functions, floating point and math

- **Device functions: inline everything** that can be inlined. A function that cannot (recursive, called through a pointer or a vtable) gets a W-lane version taking the lane mask and the arguments as vectors; a call through a pointer runs once per distinct target among the active lanes (docs/optimization/iteration-3/28-function-calls.md).
- **Floating point**: default equal to nvcc (**FMA on**; Zen 4 has FMA), with a flag equivalent to `-fmad=false`. Goal: within the ULP documented by NVIDIA, not bit-for-bit identical. The samples use `-fmad=false` for exact comparison.
- **Own vector math library** (without SLEEF), written in IR/intrinsics to be inlined into the kernel (not a separate `.s`, which would require a calling convention and saving zmm). Validate against the RTX 5060.
- `half`/`bf16` and any type without native support: emulate (convert to float). Left for later.

### 5.7 Warp intrinsics

First: `__shfl_sync` and variants (permutes between the 2 halves, `vpermt2ps`), `__ballot_sync` (the mask itself), `__any_sync`/`__all_sync`, `__activemask`, `__popc`, `__clz`, `__syncwarp` (no-op). `__match_any_sync` via `vpconflictd`. **Cooperative groups last.**

### 5.8 Atomics

- **Default / fallback**: a scalar loop only over the active lanes (`tzcnt` + `blsr` on the mask), one `lock` per lane:

  | CUDA | x86 |
  |---|---|
  | `atomicAdd` int, result ignored | `lock add` |
  | `atomicAdd` int, result used | `lock xadd` |
  | `atomicAdd` float/double | `lock cmpxchg` loop |
  | `atomicCAS` | `lock cmpxchg` |
  | `atomicExch` | `xchg` |
  | `atomicAnd/Or/Xor` without result | `lock and/or/xor` |
  | with result, `atomicMin/Max` | `lock cmpxchg` loop |

- **Planned optimization**: uniform address (global or shared) = SIMD reduction + **one** `lock xadd` + exclusive prefix sum for the return values.
- **Shared memory also needs `lock`**: the two SMT threads execute warps of the same block at the same time. No `lock` only when the block runs on a single HW thread (small blocks); the compiler may generate both variants and the runtime chooses at launch.
- `vpconflictd` for histogram-like cases, only if we measure a gain.
- CUDA atomics are *relaxed*; x86's `lock` is stronger, which is correct.

### 5.9 Memory ordering

x86 is TSO. The compiler must not reorder accesses across barriers and `__threadfence`; `volatile` is respected.

### 5.10 `printf` and `assert` on the device

High priority (without them it is hard to debug the simulator). Per-block buffer, flushed at `cudaDeviceSynchronize`, as on the GPU.

### 5.11 Unsupported features

Always a **compile error**, never a silently wrong result. This includes sm_120 features that we report (section 6.6) but do not support.

---

## 6. Driver and runtime

### 6.1 Two backends, one interface

A **device interface** (CUDA driver API style) with two implementations, **built and tested together**:

- **In-process**: workers in the application's own process.
- **Daemon**: resident process that reserves cores and VRAM; the application talks to it.

Minimum operations of the interface: open device / query properties, allocate and free, memcpy (H2D, D2H, D2D), memset, load module, launch kernel, synchronize, query error.

Every test runs on both backends. If it passes on one and fails on the other, the problem is in the transport, not in the compiler.

### 6.2 VRAM

- **Exclusive to the driver** (decided): device pointers are only valid inside the driver, as on a dedicated GPU without unified memory. All data goes through `cudaMemcpy`.
- **Allocator with pools**, prepared for later:
  ```text
  Device
  ├── pool "private"  (vram_private)  mapped only in the driver  ← POC
  └── pool "shared"   (vram_shared)   memfd at the SAME virtual   ← future
                                      address in driver and client
  ```
- Backing with hugepages (2 MB or 1 GB; helps the TLB on gathers). `MFD_HUGETLB` requires `vm.nr_hugepages`; alternative is THP with `madvise`.

### 6.3 Transport (daemon)

- **Control**: Unix socket (`SOCK_SEQPACKET`) for requests. Descriptors (modules, memfd) pass through `SCM_RIGHTS`. Later: a shared-memory ring for low-latency launch.
- **memcpy**: the driver uses `process_vm_readv/writev` directly on the client's memory (one copy, the role of DMA). Ubuntu uses `ptrace_scope=1`, so the client calls `prctl(PR_SET_PTRACER, driver_pid)` at initialization. **Fallback**: a staging ring in shared memory (two copies).
- Later: dedicated copy threads in the driver (*copy engines*) so that `cudaMemcpyAsync` overlaps copy and kernel.
- **One driver per user**: the driver executes code sent by the client.
- A kernel with an invalid access may bring down the driver (there is no cheap validation for gathers). Treat the signal as an error such as `cudaErrorIllegalAddress` when possible.

### 6.4 Modules ("fatbin") and kernel ABI

- The application embeds, per module:
  - **Native `.so`** (used now). Sent to the driver as a memfd and loaded with `dlopen("/proc/self/fd/N")`.
  - **Per-thread LLVM bitcode of the kernel** (the role of PTX). Embedded from the start, used in the future to recompile at launch with specialization (`blockDim`, `gridDim` as constants) and a disk cache.
- **Stable contract** between generated code and runtime (define early, it is what costs most to change):
  - **ABI version**: the runtime refuses an incompatible module.
  - **Per-kernel metadata**: name, size/alignment of each parameter, static shared memory, per-warp frame size (live values), number of states.
  - **Entry point**: executes **one pair of warps** from the saved state up to the next barrier or the end, and returns the reason (`BARRIER` / `EXIT`). It receives a context with `blockIdx`, dimensions, the shared memory base, the pair's frames, masks and parameters. The scheduling loop (section 4.3) stays in the runtime.
- Before the compiler exists, **2 or 3 hand-written kernels in this format** serve to test the runtime and driver.

### 6.5 Configuration

Provisional file `~/.config/cpusim/device.toml`:

```toml
[device]
cores        = 12        # "SMs" = reserved physical cores (with both SMT threads)
vram_private = "16GiB"
vram_shared  = "0GiB"    # > 0 enables the shared pool (future)
hugepages    = "2MiB"    # 2MiB, 1GiB or none
```

### 6.6 Device properties

- Base: **RTX 5060 (compute capability 12.0)**, configurable in the driver. To obtain the values, run a query program once with the real CUDA and save the fields of `cudaGetDeviceProperties` (limits of threads per block, shared memory, etc.).
- `multiProcessorCount` = configured cores; `totalGlobalMem` = `vram_private`; `warpSize` = 32.
- With `vram_shared = 0` (now): `integrated = 0`, `canMapHostMemory = 0`, `managedMemory = 0`, `concurrentManagedAccess = 0`. `cudaMallocManaged` and mapped `cudaHostAlloc` return `cudaErrorNotSupported`.

### 6.7 Profiling

- For each kernel, `cpusim-devc` generates `__cpusim_entry_<name>` and an instrumented twin `__cpusim_prof_<name>`. In the instrumented one, each executed basic block adds `popcount(mask) x the block's static cost` to per-worker counters (`ctx->counters`): fp32/fp64 flops (FMA counts 2), special operations, integer operations, loads/stores and bytes, branches, block executions and active lanes (SIMD efficiency). The counts are per thread, as in GPU profilers, and measured on the per-thread code, before vectorization.
- `CPUSIM_PROFILE=1`: the first launch of each (kernel, grid, block) runs instrumented; the following ones run the normal version and the driver returns the kernel time. The report combines counts and time (GFLOP/s, GB/s, arithmetic intensity). Each launch runs only once, so kernels that accumulate results remain correct. For data-dependent kernels, the counts are those of the first launch.
- `cpu-smi -q` shows launched and instrumented kernels.

### 6.8 Runtime API

First surface (the same one the samples use): `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaGetLastError`, `cudaGetErrorString`, launch (`<<<>>>`). Streams: everything serialized on the default stream at the beginning.

---

## 7. Unified memory (future)

When `vram_shared > 0`, the device presents itself as **integrated**, like an integrated GPU with dedicated memory plus shared memory:

- Shared pool = memfd mapped at the **same virtual address** in the driver and in the client. The client reserves early with `mmap(MAP_FIXED_NOREPLACE)`; if it collides, that client falls back to copy-only mode.
- `cudaMallocManaged`, `cudaHostAlloc` = zero-copy in that region. `cudaMemPrefetchAsync`/`cudaMemAdvise` become no-ops.
- Coherence comes for free (it is the same CPU); launch and sync signals with release/acquire semantics.
- `__device__`/`__managed__` globals placed by the compiler in the shared region.
- Optional: replace the client's `malloc`/`new` so the whole heap is shared (HMM style). It breaks programs that use `fork()` expecting copy-on-write, which is why it is a flag.
- Ordinary host stack and globals are never shareable between processes. Only the in-process mode gives full SVM.

---

## 8. Build order

1. **Driver base**: device interface, configuration, allocation (pools), communication (socket, memcpy via `process_vm_*`), module loading and launch. Daemon and in-process backends together.
2. **Execution runtime**: worker pool pinned to SMT siblings, block and pair scheduling, 2-participant barrier with `MWAITX`, frames. Tested with the hand-written kernels (section 6.4).
3. **Minimal `libcudart`** on top of the device interface.
4. **Compiler**, developed testing together with the driver: front end with our headers, classification, linearization, widening, memory, state machine.
5. POC samples passing on both backends.

---

## 9. Proof of concept (first milestone)

Minimal vertical slice, end to end:

| Layer | In the POC | Left for later |
|---|---|---|
| Front end | Clang + our minimal headers | Full API |
| Compiler | uniform/affine/varying, `if` and loops with masks, warp 32, load/store/gather/scatter, `__syncthreads`, K=1 | K=2, memory yield, warp intrinsics, atomics, math |
| Runtime and driver | `cudaMalloc/Memcpy/Memset/Free`, launch, sync, on both backends; warp pairs on SMT | streams, `cpu-smi`, full config |
| Target samples | `matmul`, `transpose`, `heat2d`, `scan`, `mandelbrot`, `bitonic_sort` | `nbody` (math), `reduction` and `spmv` (shuffle), `histogram` (atomics) |

**Acceptance criterion**: the 6 samples, with no change to the source, compiled by our compiler, print `PASS` on both backends.

---

## 10. Backlog (after the POC, roughly in this order)

- K=2 (pair interleaving)
- Warp intrinsics, atomics with aggregation, own math library → `nbody`, `reduction`, `spmv`, `histogram`
- `printf`/`assert` on the device (high priority, may come earlier if debugging demands it)
- Dynamic shared memory (`extern __shared__`)
- Profiler: time per kernel and **SIMD efficiency** (active lanes / total lanes)
- `cpu-smi` (occupancy, private and shared VRAM, processes)
- Streams with concurrent execution; several processes in the driver (single queue first, then core splitting or time slicing)
- Deterministic mode (fixed block order)
- Shared VRAM pool (section 7)
- JIT with specialization at launch from the embedded bitcode
- Memory yield with prefetch
- Emulation of `half`/`bf16` and other unsupported types
- Non-inlined device functions, recursion, function pointers
- Cooperative groups (last)

## 11. Out of scope

Thrust, CUB, cuBLAS and other libraries; inline PTX (`asm()`); Tensor Cores; dynamic parallelism; textures; project and tool names (for now).

## 12. Risks and points to verify first

- **Clang with `-nocudainc -nocudalib` and our own headers**: confirm right at the start by compiling `samples/matmul.cu` (separating host and device, obtaining the per-thread device IR).
- Gather/scatter performance on Zen 4 (scatter is microcoded): measure early, it determines how much the affine classification matters.
- Cost of the state switch at barriers and of the barrier between SMT siblings.
- `process_vm_readv` with `ptrace_scope` and `PR_SET_PTRACER`.

## 13. References

ISPC; Region Vectorizer / Whole-Function Vectorization (Karrenberg, Moll, Hack); GPU Ocelot (PTX to x86); MCUDA; CuPBoP; COX (warp functions on CPU); AdaptiveCpp (barriers on CPU by compiler); POCL; AMAC / interleaving with coroutines (for memory yield); ZLUDA (reimplementation of the CUDA API).
