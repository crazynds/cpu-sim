# CudaSimulator: design

How the project is built and why: the choices that hold today, at the level of the whole system. How each piece was made fast, with measurements, is in [docs/optimization/](docs/optimization/README.md), one document per optimization; this file points to them instead of repeating them. Ideas that were tried and dropped are recorded in [22](docs/optimization/iteration-2/22-what-did-not-work.md) and are not repeated here.

Names (`cucc`, `cpusimd`, `cpu-smi`) are still provisional.

---

## 1. Goal

Take a CUDA program that works on an NVIDIA GPU and, **without changing the source code**, compile it to run on the CPU, with the threads of a block executed on AVX-512 vector lanes.

- **Ahead-of-time compilation, not interpretation.** The kernel, written per thread, becomes native code that runs a group of threads in SIMD. Scheduling, barriers and block control live in the generated code and in a runtime of our own.
- **A resident driver** plays the role of the NVIDIA driver and the GPU: it reserves cores and memory ("VRAM"), and receives modules, parameters and launches from every program of the user.
- **Nothing from NVIDIA is used or distributed**: the CUDA headers, the runtime and the driver are ours. The simulated CUDA version is **13**.
- **AVX-512 is required.** Without it the tools stop with a clear message; there is no fallback.

---

## 2. Status

| Area | State |
|---|---|
| Compiler | the CUDA used by real programs: divergence, barriers, static and dynamic shared memory, per-thread arrays, warp intrinsics, atomics, device and constant variables, recursion and function pointers, `printf`/`malloc`/`assert`, cooperative groups, libcu++ atomics, `half`/`bfloat16`, `wmma` tensor cores |
| Runtime | memory and copies (linear, pitched, 3D, pinned, mapped, managed), asynchronous streams and events, CUDA graphs, occupancy and attribute queries, error codes with sticky errors |
| Driver | `cpusimd` serving several programs at once, `cpu-smi`, kernel faults and hangs isolated per program |
| Validation | `ctest` (features at every width and on both backends, samples, bit-for-bit stability); the stress suite, 631 cases checked against the GPU, 629 passing ([29](docs/optimization/iteration-3/29-stress-suite.md)) |
| Performance | against the same computation in OpenMP on the same CPU, 335 of 541 cases faster, geometric mean 1.24x ([30](docs/optimization/iteration-3/30-launches-and-hot-spots.md)) |

### Development machine

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 8940HX (Zen 4), 16 cores / 32 threads, 2 CCDs |
| AVX-512 | F, DQ, BW, VL, CD, VNNI, BF16, IFMA, VBMI, VBMI2, BITALG, VPOPCNTDQ. **No FP16.** 512-bit operations run as two 256-bit halves |
| Caches | L1d 32 KB/core, L2 1 MB/core, L3 2 x 32 MB |
| Reference GPU | NVIDIA GeForce RTX 5060 Laptop (compute capability 12.0) |
| CUDA SDK | 13.0 at `/usr/local/cuda-13.0`, used **only** to run the same programs on the real GPU as an oracle |
| Toolchain | GCC 13, CMake 3.28, LLVM/Clang 19 |

---

## 3. Overview

```text
                     program.cu (unmodified)
                              │
                 ┌────────────┴─────────────┐
                 │ cucc                     │  Clang as the C++/CUDA parser,
                 │                          │  our CUDA 13 headers
                 └────────────┬─────────────┘
          host code           │          device code (per-thread LLVM IR)
              │               │                   │
              │               │        cpusim-devc: SIMT pass, W-lane variants,
              │               │        barriers, memory, calls, atomics
              │               │                   │
              ▼               │                   ▼
       program binary ◄───────┴──── fatbin: generic .so, one .so per block
              │                     shape known at compile time, bitcode
              │ links with libcpusim_cudart
              ▼
     ┌───────────────────────── device interface ─────────────────────────┐
     │  in-process backend              │  daemon backend (cpusimd)       │
     │  workers inside the program      │  socket + the driver's memory   │
     └────────────────────────────────────────────────────────────────────┘
                                         │
                                      cpu-smi
```

At the first launch of a block shape that the binary does not carry, the runtime compiles the bitcode for that shape (JIT) and keeps the result in a disk cache.

---

## 4. Execution model

### 4.1 Warps and groups of lanes

- **A warp is 32 threads semantically** (`warpSize == 32`, lane ids, warp intrinsics, `__activemask`), because a lot of real code depends on it. `CPUSIM_WARP_SIZE=16` or `64` at compile time changes it for a whole program: 16 fills exactly one register (finer divergence and barrier groups, for code written against `warpSize`), 64 is AMD's wavefront (CUDA's masks are 32-bit, so `__ballot_sync` and `__activemask` then cover lanes 0..31; `__any`, `__all` and cooperative groups use all 64). Tensor cores need 32 ([32](docs/optimization/iteration-3/32-warp-size.md)).
- Physically, one thread is one 32-bit lane of a zmm register, so a vector holds 16 threads and a 32-thread value takes two registers; the mask of active threads lives in the `k` registers.
- A block smaller than a warp runs in fewer lanes: with its shape known (section 5.2), a block of up to 16 threads has a 16-lane warp entry (up to 32 with warps of 64), with the warp's semantics unchanged; the lanes left out are threads that do not exist.
- **The unit of execution is a group of W lanes**: one call of a generated function runs W threads of one block in lockstep, with W from 32 to 512 (section 4.3). The warp's semantics (shuffles, votes) stay per 32 lanes inside the group.
- There is no independent thread scheduling (Volta and later): the lanes of a group advance together. A lane that waits for another lane of its own warp hangs, as on GPUs before Volta (section 4.7 covers waits between warps).

### 4.2 Workers and blocks

- The device reserves physical cores (by default all but two, which stay with the programs and the system); each of their logical CPUs gets one **worker** thread pinned to it. A core is an "SM", so `multiProcessorCount` is the number of reserved cores.
- **A worker runs whole blocks by itself.** The two SMT siblings of a core normally run different blocks, and the shared memory and warp frames of a block live in a worker-local buffer that stays in L1/L2.
- The original plan was the opposite: every block on a whole core, its warps split between the two siblings. Measured, it loses whenever the kernel alternates memory phases and compute phases (tiled matmul): independent blocks on the siblings fall out of phase and SMT overlaps one's memory wait with the other's arithmetic, while a shared block aligns both on the same phase. Running one block across both siblings was kept for a while as a **cooperative variant** of the width selection ([15](docs/optimization/iteration-2/15-smt-cooperative-variant.md)); measured over the tracked HeCBench kernels it never won, and was from 2 to 50 times slower than the best width, so it was removed.

### 4.3 Width variants

For each kernel the compiler generates several functions; the runtime picks one per (kernel, block shape).

| Variant | Lanes per call | What happens at `__syncthreads` | Valid for |
|---|---|---|---|
| warp (`32`) | the warp size (fewer for a block smaller than a warp) | the warp saves its live values in a frame and returns; the executor runs the block's other warps up to the barrier, then resumes each one | every kernel |
| suspending (`64s`, `128s`) | 2 or 4 warps | the same, for groups of 2 or 4 warps (fewer switches) | kernels with barriers, blocks larger than the group |
| wide (`64` to `512`) | 2 warps up to 512 | nothing: the whole block is in one call, in lockstep | without barriers: any block of at least W threads; with barriers: the smallest W that covers the block |

(The names are those of warps of 32; with 16 or 64 the widths scale with the warp.)

Wider groups give the core more independent work per instruction stream; they also multiply the registers each value needs, and past the 32 zmm the spills cost more than the gain. No width wins everywhere, which is why it is measured ([02](docs/optimization/iteration-1/02-width-variants.md)).

### 4.4 Choosing the width

The runtime measures the valid candidates and keeps the fastest, per kernel and block shape, until the program ends ([03](docs/optimization/iteration-2/03-automatic-width-selection.md)):

- **Large launches** measure inside the first launch: each candidate runs a slice of the launch's own blocks (32 per worker by default, `CPUSIM_TRIAL_BLOCKS`), in two rounds back to back keeping each candidate's best slice, and the rest of the launch runs with the winner. The measure is the workers' time per block, not the wall time of the slice.
- **Small launches** measure one candidate per launch, by wall time per block. After the first round only candidates within 1.5x of the best are run again, so a program that launches a shape a few times does not spend them all measuring.
- **Order of the candidates.** With barriers: wide, then suspending, and the 32-lane variant last, since a program that ends before the measurement does keeps the first ones, and the 32-lane variant, which suspends at every barrier, was the slowest on every barrier kernel measured.
- `CPUSIM_LANES=<n>`, `<n>s` or `<n>r` (lanes that refill) forces a width; `CPUSIM_VERBOSE=1` prints the measurements and the choice.

Every width produces the same bits: a test suite checks it, with the selection switching widths in the middle of a program ([23](docs/optimization/iteration-3/23-reproducibility-and-stability.md)).

### 4.5 Barriers

- In a wide variant `__syncthreads` disappears: the block is one call.
- In the warp and suspending variants the kernel becomes a state machine: each barrier is a resume point, the group saves only the values live after it, and a dispatcher at the function entry restores them and jumps there. The last group to arrive goes through without saving anything (`barrier_pass`) ([09](docs/optimization/iteration-1/09-state-at-barriers.md), [10](docs/optimization/iteration-2/10-barrier-pass.md)).
- Threads that returned do not count toward the barrier, as in CUDA.
- `__syncthreads_count/_and/_or` are a counter in shared memory around an ordinary barrier.
- The grid barrier of cooperative groups works in cooperative launches, which may not have more blocks than the device runs at once.

### 4.6 Divergence

- Control flow that may differ between lanes is linearized: each basic block runs under a mask, and a block whose mask is empty is skipped.
- Loops run while some lane is active; lanes leave the mask as they exit.
- What the compiler proves uniform keeps real control flow: uniform branches and uniform loops are plain CPU branches ([07](docs/optimization/iteration-1/07-uniform-loops.md)).
- Small divergent `if`s become selects, with the loads inside them guarded by the mask ([08](docs/optimization/iteration-1/08-if-conversion.md)).
- An inactive lane keeps its values (merge masking) and never touches memory: masked loads, stores, gathers and scatters do not fault on inactive lanes.

### 4.7 Waiting on other threads

A loop whose exit depends on a volatile or atomic read (a spin lock, a flag set by another block or warp) gets a **yield point**: the group returns `CPUSIM_YIELD`, the executor runs the block's other groups, then calls it again. Kernels with such loops keep to the 32-lane variant, and the waiting side of a branch is emitted after its siblings, so the thread that will write runs before the one that waits.

### 4.8 Handing out blocks

- **Distribution** ([17](docs/optimization/iteration-2/17-block-distribution.md)): the grid is dealt in chunks of 64 blocks, chunk c to worker c mod n, so the same block lands on the same core launch after launch and finds its data in cache; a worker that runs out steals from the others. A grid too small for a chunk of 64 per worker gets one chunk per worker.
- **Order** ([04](docs/optimization/iteration-1/04-block-order.md)): 2D grids are walked in bands of 8 columns, so blocks that run at the same time share their working set in L3 (`CPUSIM_BLOCK_ORDER=linear` turns it off).
- **Few blocks** ([16](docs/optimization/iteration-2/16-splitting-blocks-into-warps.md)): when a launch has fewer than 8 blocks per worker and the kernel has no barrier and no shared memory, the executor hands out groups of lanes instead of blocks, as a GPU schedules warps.
- **Concurrent kernels** ([20](docs/optimization/iteration-2/20-concurrent-kernels.md)): several launches (from different streams or programs) can be active; a worker takes blocks from the oldest one with blocks left, so the end of one kernel overlaps the start of the next. 
- **Idle workers** ([30](docs/optimization/iteration-3/30-launches-and-hot-spots.md)): they spin for up to 200 us (on Zen, parked with `MONITORX`/`MWAITX`, which leaves the core to the sibling), then sleep. A launch posts how many workers it needs, `min(blocks, workers)`, so a one-block kernel wakes one worker, and workers join a launch without a lock. An empty launch costs about 2 us in-process and 5 to 7 us through the driver.

### 4.9 Faults

A kernel that makes an invalid access or traps does not take the driver down: each worker has a signal handler on an alternate stack that leaves the kernel, and the launch fails with `cudaErrorIllegalAddress` (or `cudaErrorAssert`), sticky in that program's context as on a GPU. A program that dies with a kernel still running has the kernel cancelled, so a hang in one program does not hang the others.

---

## 5. Compiler

### 5.1 Front end

- `cucc` drives Clang in CUDA mode with `-nocudainc -nocudalib` and **our own headers** (`cuda_runtime.h`, `vector_types.h`, the device functions, `cuda_fp16.h`, `cuda_bf16.h`, `mma.h`, `cooperative_groups.h`, `<cuda/atomic>`). A `.cu` is full C++, so writing a parser of our own was never an option.
- Host code: `<<<>>>` becomes calls into `libcpusim_cudart`; kernels and device variables register through the mechanism Clang emits.
- Device code: per-thread LLVM IR for NVPTX, retargeted to x86-64 by `cpusim-devc`.
- Floating point as nvcc: FMA contraction on by default (`-ffp-contract=on`, decided in the front end so every width rounds the same way), `-fmad=false` to turn it off. The goal is to stay within the ulp bounds NVIDIA documents, not to be bit-identical to the GPU.

### 5.2 Modules and specialization

The fatbin carries, per translation unit, a **generic module** (any block shape), one **specialized module** per block shape that is a constant at some launch in the source (`k<<<grid, 128>>>`, compiled in parallel by `cucc`), and the **per-thread bitcode**, which plays the role of PTX. A launch with a shape the binary does not carry is compiled from the bitcode at that launch and cached in `~/.cache/cpusim/jit`; `CPUSIM_JIT=0` uses the generic module instead.

Specializing on the block shape is what made the wide variants viable: `blockDim` becomes a constant and `threadIdx` a table of known values, so the compiler can prove at compile time which accesses are contiguous ([01](docs/optimization/iteration-1/01-specialized-jit.md), [12](docs/optimization/iteration-2/12-static-access-analysis.md)). To keep the build fast, the generic module drops the wide variants that are too large and leaves them to the JIT ([26](docs/optimization/iteration-3/26-compiler-and-runtime-changes.md)).

### 5.3 The SIMT pass

`cpusim-devc` turns the per-thread function into the W-lane variants of section 4.3. LLVM's auto-vectorizer is not used (it does not vectorize a whole function); the generated IR is explicitly vector (`<W x T>` values, masked loads and stores, gathers and scatters), and register allocation and scheduling are LLVM's, followed by `-O3` for `znver4`.

1. **Uniformity analysis** (ours). Values are uniform across the group (`blockIdx`, parameters, a loop counter all lanes share), varying (`threadIdx`, loads through varying pointers, values that join divergent paths), or affine in the lane. Uniform values stay scalar.
2. **Linearization** with masks (section 4.6).
3. **Widening** of every varying value to W lanes.
4. **Memory.** An address is split into a uniform base and a varying offset, so that most of its arithmetic is scalar and the access patterns can be tested on the offsets alone ([05](docs/optimization/iteration-1/05-address-decomposition.md)). Then, per 16-lane chunk: proven contiguous at compile time, a contiguous masked load or store; proven equal, a broadcast; otherwise a run-time test of the offsets picks contiguous, broadcast or gather ([06](docs/optimization/iteration-1/06-dynamic-coalescing.md)). Bytes and 16-bit values are gathered as the 32-bit words holding them. Gathers and scatters that remain take 32-bit indices counted from the access's root pointer: an access stays inside its object, so the index fits whenever the object is under 2 GiB, which is always true of shared memory and per-thread arrays and, for global memory, true unless the root points into an allocation of 2 GiB or more (the allocators keep those at the top, the runtime passes their address span in the context and the kernel compares the root with it; device `malloc` refuses objects that large). Gathers and especially scatters are slow on Zen 4 (about 0.7 cycle per element), so the point of all of this is to avoid them ([31](docs/optimization/iteration-3/31-gathers-and-scatters.md)).
5. **Barriers and yields**: the state machine of section 4.5, the yield points of 4.7.
6. **Specialized arithmetic**: division by a uniform divisor becomes a multiplication by a magic number ([14](docs/optimization/iteration-2/14-magic-number-division.md)); `blockIdx` carries CUDA's range limits, so 32-bit index arithmetic can be proved not to wrap ([13](docs/optimization/iteration-2/13-no-overflow-proof.md)); strided accesses inside loops get software prefetches ([19](docs/optimization/iteration-2/19-automatic-prefetch.md)).

### 5.4 Per-thread memory

- **Per-thread arrays** (`float acc[4]`) live in the group's stack frame lane-major (`acc[i][lane]`), like NVIDIA's local memory: a uniform index is one contiguous access. An array whose address escapes (passed to a recursive function) gets W real copies instead.
- **Shared memory** is a uniform base in the call's context plus an offset; static size from the kernel's metadata, dynamic size from the launch.
- **Device, constant and managed variables** live in the VRAM, one copy per program, reached from every module (generic and specialized) through a pointer slot the runtime fills when it loads the module.

### 5.5 Functions

- Device functions are **inlined** wherever possible.
- A function that cannot be (recursive, called through a pointer or a virtual call) gets a **W-lane version** taking the group's mask and every argument as a vector. A recursion ends because the block holding the call runs only while some lane is active; a call through a pointer runs once per distinct target among the active lanes; a function's address is a stable ID, so pointers stored in device memory work across modules ([28](docs/optimization/iteration-3/28-function-calls.md)).
- Calls into the host process (`printf`, `malloc`, `free`, `assert`) run once per active lane; `printf` writes to the program's standard output.

### 5.6 Atomics and ordering

- An atomic whose address is the same in every active lane is one reduction across lanes, **one** locked instruction, and an exclusive scan for the values returned.
- Otherwise lanes are grouped by address: one locked instruction per distinct address, its operand reduced over the lanes that share it. CUDA leaves the order of a warp's atomics unspecified, so this is an order a GPU could choose ([30](docs/optimization/iteration-3/30-launches-and-hot-spots.md)).
- Atomics in shared memory are locked too.
- x86 is TSO and its locked instructions are stronger than CUDA's relaxed atomics. The compiler keeps `volatile` accesses, fences and barriers in order with compiler barriers.

### 5.7 Math, types and warp operations

- **Math**: AVX-512 instructions where they exist (`sqrt`, `fma`, rounding, min/max); transcendental single-precision functions are computed in double with glibc's vector library (libmvec) and rounded once, double precision with glibc's scalar functions per lane, since libmvec errs by up to 4 ulp where CUDA promises 1 or 2. Integer division never faults: divisors of zero (and `INT_MIN / -1`) are replaced, as the GPU returns an unspecified value.
- **`half` and `bfloat16`**: each operation is done in `float` and rounded once, which gives the GPU's correctly rounded result ([24](docs/optimization/iteration-3/24-16-bit-types.md)).
- **Tensor cores** (`nvcuda::wmma`, 16x16x16): one per-warp operation on `vdpbf16ps`, `vpdpbusd` or F16C with FMA; bfloat16 is emitted inline with the accumulators in registers. The device advertises the compute capability these instructions can honor, and refuses kernels that ask for more ([25](docs/optimization/iteration-3/25-tensor-cores.md), [27](docs/optimization/iteration-3/27-kernel-at-peak.md)).
- **Warp intrinsics**: shuffles are permutes within and between the two halves of a warp, `__ballot_sync` is the mask itself, `__match_any_sync` uses `vpconflictd`, `__reduce_*_sync` reduces across lanes; cooperative groups are headers over these.

### 5.8 Unsupported features

Always a **compile error**, never a silently wrong result.

---

## 6. Driver and runtime

### 6.1 Two backends, one interface

A device interface in the style of the CUDA driver API has two implementations, built and tested together:

- **daemon**: `cpusimd`, a resident process per user that reserves the cores and the VRAM and serves every program;
- **in-process**: the same executor inside the program.

`CPUSIM_BACKEND=auto` (the default) uses the driver when it is running. Every feature test runs on both: a difference between them is a transport bug, not a compiler bug.

### 6.2 Memory

- **Private VRAM**: device pointers are valid only in the driver, as on a discrete GPU, and data moves through copies. Backed by 2 MiB (or 1 GiB) huge pages, which helps the TLB on gathers; `cudaMalloc` materializes its pages before returning, so a kernel does not stall on page faults ([18](docs/optimization/iteration-2/18-cudamalloc-pages.md)).
- **Shared pool** (`vram_shared`, 4 GiB of address space by default): a memory file mapped at the **same virtual address** in the driver and in every program, so a pointer into it works on both sides. `cudaMallocManaged` and mapped `cudaHostAlloc` allocate there, without copies. `__managed__` variables and memory registered with `cudaHostRegister` are kept equal to the host's copy around each launch. With the pool, the device reports itself as integrated (`managedMemory`, `canMapHostMemory`).

### 6.3 Transport

- **Control**: a Unix `SOCK_SEQPACKET` socket; modules and memory files pass as descriptors (`SCM_RIGHTS`). The program also passes its standard output, for device `printf`.
- **Copies**: the driver reads and writes the program's memory directly with `process_vm_readv/writev` (one copy, the role of DMA); the program allows it with `prctl(PR_SET_PTRACER)`.
- Both ends of the socket spin briefly before sleeping, which keeps a launch round trip at a few microseconds.
- The driver checks every second for programs that are gone, cancels their kernels and frees their memory.

### 6.4 Kernel ABI

The contract between generated code and runtime is `src/device/kernel_abi.h`, versioned (`CPUSIM_KERNEL_ABI`, now 13); the runtime refuses a module of another version. A module exports, per kernel:

- the entry of each variant (`__cpusim_entry_<k>`, `__cpusim_entry<W>_<k>`, `<W>s`, `<W>c`): it runs one group from its saved state to the next barrier, yield or the end, and returns the reason;
- an instrumented twin for profiling (`__cpusim_prof_<k>`);
- metadata (`__cpusim_meta_<k>`): parameter layout, static shared memory, frame size, which variants exist, flags (barriers, waiting loops), the warp size and the warp entry's lanes.

Per module, the pointer slots and initial values of device variables. Each call receives a context with the block's coordinates, dimensions, shared memory base, frame, per-lane thread indices and the address span of the allocations of 2 GiB or more.

### 6.5 Runtime API

The runtime implements the part of the CUDA API that real programs use: devices and properties, every form of memory and copy, asynchronous streams (one queue and thread per stream, default-stream rules, events, `cudaStreamWaitEvent`, host callbacks), CUDA graphs (stream capture and explicit nodes), symbols, function attributes and occupancy, cooperative launches, error handling with sticky errors.

### 6.6 Configuration and properties

`~/.config/cpusim/device.toml` (`cpusimd --init-config` writes one with this machine's defaults):

```toml
[device]
name         = "My CPU GPU"
cores        = 14        # physical cores to reserve ("SMs"); or cpus = "2-15,18-31"
vram_private = "16GiB"
vram_shared  = "4GiB"    # shared pool; "0" disables it
hugepages    = "2MiB"    # 2MiB, 1GiB or none
prefault     = false     # true: occupy all of the VRAM at startup

[driver]
socket = "/run/user/1000/cpusim/driver.sock"
```

Device properties follow an RTX-class GPU (thread, block and shared memory limits), with `multiProcessorCount` = reserved cores, `totalGlobalMem` = private VRAM, `warpSize` = 32, and the compute capability from the tensor core support above.

### 6.7 Tools

- **`cpu-smi`**: like `nvidia-smi`: utilization (time with a kernel running), real load of the reserved cores, memory, processes, temperatures and fans.
- **Profiling** (`CPUSIM_PROFILE=1`): the first launch of each (kernel, grid, block) runs the instrumented twin, which counts per basic block flops, integer operations, memory operations and bytes, branches and active lanes (SIMD efficiency); later ones are timed. The report combines both.
- **Debug mode** (`CPUSIM_DEBUG=1`): one group at a time on one thread, shared memory and frames between guard pages; the first invalid access is reported with block, warp, address, region and instruction.
- **`CPUSIM_DUMP_ASM=1`**: the x86 assembly of each launched kernel, per block shape.

These are read by the program and travel with each launch, so the one driver runs each launch in the mode its program asked for.

---

## 7. Limits and out of scope

- **Out of scope**: Thrust, CUB, cuBLAS and other NVIDIA libraries; inline PTX (`asm()`); dynamic parallelism; textures and surfaces.
- **Semantics**: no independent thread scheduling inside a warp (section 4.1); with warps of 64, CUDA's 32-bit warp masks cover only lanes 0..31; managed and mapped memory are coherent with the host at kernel boundaries, not while a kernel runs; a cooperative launch is limited to two blocks per core.
- **Performance** (the slow cases are listed in [30](docs/optimization/iteration-3/30-launches-and-hot-spots.md)): atomics contended by every core on the same cache lines, kernels tuned for a GPU's memory system, deep per-thread divergence, one-byte-per-thread code (16 bytes per vector where a CPU loop handles 64), and the socket round trip per launch through the driver.

## 8. Open directions

Not tried yet:

- Sending launches to the driver without waiting for each reply, or through a shared-memory ring, to remove most of the per-launch round trip.
- A memory yield: in chains of dependent loads, prefetch and switch to another group, with the same state machine as the barriers.
- The block distribution policy as one more dimension of the automatic selection ([17](docs/optimization/iteration-2/17-block-distribution.md)).

## 9. References

ISPC; Region Vectorizer and Whole-Function Vectorization (Karrenberg, Moll, Hack); GPU Ocelot (PTX to x86); MCUDA; CuPBoP; COX (warp functions on the CPU); AdaptiveCpp (barriers on the CPU by the compiler); POCL; AMAC and interleaving with coroutines; ZLUDA (a reimplementation of the CUDA API).
