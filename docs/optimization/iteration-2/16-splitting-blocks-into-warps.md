# 16. Splitting blocks into warps

> **In one sentence:** when a kernel launches few blocks and those blocks do not need to communicate internally (no barrier, no shared memory), the executor stops handing the whole block to a single worker and instead hands each group of lanes of the block out as a separate task, the same way a GPU schedules warps; this way all 28 workers work instead of only 10.

The project's general concepts (group of lanes, worker, warp context) are in [00-basic-concepts.md](../00-basic-concepts.md). The essentials here: the threads of a block are grouped into **groups of lanes** (normally 32, the size of a warp), each group is one call of the generated function, and the 28 workers are OS threads pinned to CPUs 2 to 15 and 18 to 31.

## The problem

Until now the executor's scheduling unit was the block: each block went whole to a single worker, which executed its groups in sequence. With grids of hundreds of blocks this saturates the 28 workers. With a grid of 10 blocks, at most 10 workers have work, and the total time becomes that of the most expensive block (the makespan is bounded by the largest indivisible task).

This is what happened with `matrix-rotate`, from HeCBench (a public collection of CUDA programs used as a benchmark).

The kernel rotates a 5000 x 5000 matrix by concentric layers (rings from the edge toward the center):

```text
layer 0 (outer edge)          layer 1              ...   layer 2499 (core)
+-----------------+      +-----------------+
|#################|      |                 |
|#               #|      |  #############  |
|#               #|      |  #           #  |
|#               #|      |  #           #  |
|#################|      |  #############  |
+-----------------+      +-----------------+
 ~5000 elements           a little fewer          1 element
 per side                                           
```

There are 2500 layers, and the program uses **one thread per layer**. It launches blocks of 256 threads, so the grid has:

```text
(2500 + 255) / 256 = 10 blocks
```

An important detail: the thread of layer 0 traverses about 5000 elements per side; the one of the last layer traverses 1. The work is very uneven, and whoever has the outer layers (block 0) works the most.

The executor, before this optimization, gave **one whole block per worker**. Result:

| What | Value |
|---|---|
| Blocks in the grid | 10 |
| Available workers | 28 |
| Workers with work | 10 |
| CPUs in use, on average (measured) | 2.5 |
| Share of the total work in block 0 | about 19%, all on a single core |

The "2.5 CPUs on average" shows that not even the 10 workers were busy all the time: the inner blocks (small layers) finished quickly, and in the end only the worker of block 0 kept working, alone.

## The idea

On a GPU, the unit that the hardware schedules is not the block, it is the **warp**: the block is resident on an SM (which guarantees common shared memory and `__syncthreads()`), but the SM's warp schedulers issue instructions from any ready warp, with no fixed order among the warps of the block. The co-residence of the block exists only to sustain shared memory and barriers.

Therefore, if a kernel **has no barrier** and **does not use shared memory**, the warps of the same block have no communication channel exclusive to the block. What remains is global memory, which is visible to any thread of the grid and for which CUDA does not guarantee ordering between warps without barriers or atomics; a correct program, therefore, cannot already depend on the warps of the block running together. In that case each group can be treated as an independent task and scheduled on any worker.

For `matrix-rotate`:

```text
before: 10 blocks            -> 10 tasks  (at most 10 workers)
after:  10 blocks x 8 groups -> 80 tasks  (all 28 workers work)
```

And block 0, which concentrated the longest layers, now becomes 8 tasks spread over 8 different workers.

## How it works, step by step

Reduced example: a grid of 2 blocks, each with 4 groups (G0 to G3), and 3 workers.

**Without splitting** (unit = block):

```text
unit 0 = block 0 (G0 G1 G2 G3)   -> worker A runs the 4 groups in sequence
unit 1 = block 1 (G0 G1 G2 G3)   -> worker B runs the 4 groups in sequence
worker C: nothing to do
```

**With splitting** (`split = 4`, unit = group of a block):

```text
unit u  ->  block = u / 4,  group = u % 4

u=0 -> block 0, G0      u=4 -> block 1, G0
u=1 -> block 0, G1      u=5 -> block 1, G1
u=2 -> block 0, G2      u=6 -> block 1, G2
u=3 -> block 0, G3      u=7 -> block 1, G3
```

Now there are 8 units, and the 3 workers can share them. The rest of the executor (the way units are distributed, described in [17-block-distribution.md](17-block-distribution.md)) changes nothing: it just starts counting "units" instead of "blocks".

The concrete steps are:

1. In `Executor::run`, before creating the work ranges, the executor tests whether the launch is eligible (details in the "When it applies" section).
2. If so, it stores in `job.split` the number of groups per block and multiplies the total number of units (`job.blocks`) and the starting point (`first`) by this number.
3. Each worker, in `run_blocks`, takes a unit `u`, finds the block (`u / split`) and the group (`u % split`), fills in the block coordinates (`blockIdx`) in the context of that group and calls the generated function once.

Each worker already keeps the contexts (`cpusim_warp_ctx`) of all the groups of the block, with the `threadIdx` of each lane precomputed. Executing only group 5 is selecting context 5, overwriting `blockIdx` and calling the function; no additional state needs to be created.

## The code

### The decision to split

`src/device/executor.cpp:346-355`

```cpp
    // The warps of a block that has no barrier and no shared memory never
    // talk to each other: with few blocks, hand out its groups one by one
    // (as a GPU schedules warps), so the work spreads over every worker
    // instead of one worker per block. matrix-rotate launches 10 blocks.
    if (!job.cooperative && !block && !(meta.flags & CPUSIM_META_HAS_BARRIERS) && cfg.shared_bytes == 0 &&
        warps.count > 1 && total < 8 * threads_.size()) {
        job.split = warps.count;
        job.blocks *= job.split;
        first *= job.split;
    }
```

The condition of the `if` requires, simultaneously:

- it is not a cooperative launch (the one in which two sibling CPUs run a block together, see [15-smt-cooperative-variant.md](15-smt-cooperative-variant.md));
- it is not using the block function (`block`), which runs the whole block at once;
- the compiler marked that the kernel **has no barriers** (`CPUSIM_META_HAS_BARRIERS` off);
- the launch **does not use** dynamic shared memory (`cfg.shared_bytes == 0`);
- the block has more than one group (`warps.count > 1`; with a single group, there is nothing to split);
- the grid is small: fewer than 8 blocks per worker (`total < 8 * threads_.size()`; with 28 workers, fewer than 224 blocks).

When all of this holds, `job.split` receives the number of groups per block, and the counts become counts of units.

### The `split` field in the Job

`src/device/executor.h:103-106`

```cpp
        // > 1: the unit handed out is one group of a block, not the block
        // (kernels without barriers or shared memory, small grids): `next`
        // and `blocks` count units, unit u = block u / split, group u % split.
        uint32_t split = 1;
```

The default value is 1, which means "the unit is the whole block", the previous behavior.

### The loop that runs one group at a time

`src/device/executor.cpp:134-143`

```cpp
    if (job.split > 1) {
        // One group of a block per unit; the groups need nothing else.
        uint64_t u;
        while (claim(job, worker, &u)) {
            cpusim_warp_ctx* c = ctx.at(size_t(u % job.split));
            block_coords(job, u / job.split, c->block_idx);
            job.entry(c);
        }
        return;
    }
```

- `claim` hands the worker the next unit that is its own (or steals one from another worker, see [17-block-distribution.md](17-block-distribution.md)).
- `ctx.at(u % job.split)` selects the context of the right group.
- `block_coords(job, u / job.split, ...)` computes the `blockIdx` of that block (respecting the block order described in [04-block-order.md](../iteration-1/04-block-order.md)).
- `job.entry(c)` calls the generated code for that group of lanes.

There is no barrier loop or state management: since the kernel has no `__syncthreads()`, each call runs the group from beginning to end.

## When it applies (and when it does not)

**Why it is safe.** The only communication mechanisms with block scope are shared memory and barriers. Without either of them, the order and the worker in which each group runs do not change the result of any correct program. Global memory remains visible to everyone, as before; the CUDA memory model does not guarantee ordering between warps or between blocks without explicit synchronization, so a program that depended on this would already have a race condition on the GPU.

**When it does not apply:**

| Situation | Why it does not split |
|---|---|
| Kernel with `__syncthreads()` | the groups need to wait for one another on the same worker |
| Launch with dynamic shared memory | each worker has its own shared area; groups on different workers would see different areas |
| Cooperative variant or block function | the block is treated as a whole by construction |
| Block with a single group (e.g., 32 threads with 32 lanes) | there is nothing to split |
| Grid with 8 or more blocks per worker | there is already plenty of work for everyone; splitting would only increase the number of calls and scatter neighboring blocks |

On the last row: with a large grid, handing out whole blocks has advantages. The groups of a block tend to access neighboring cache lines (and sometimes the same ones), and executing them in sequence on the same core reuses what is already in L1/L2; spreading them across different cores duplicates those lines in several L2s and, across CCDs, forces them to be fetched again. For this reason the split is only turned on when work would be lacking.

A note on shared memory: the check uses `cfg.shared_bytes`, which in the launch is the total for the block (static plus dynamic, as the comment of `Executor::run` in `executor.h` says). Kernels with static shared memory normally also have barriers, which would already block the split.

## Results

| Program | Before | After | Reference |
|---|---|---|---|
| matrix-rotate | 18.1 ms | ~5.5 ms (3.3x faster) | GPU: 6.48 ms |
| nbody | 770 GFLOP/s | 885 GFLOP/s | |

On `matrix-rotate`, the simulator on the CPU became **faster than the reference GPU** (5.5 ms versus 6.48 ms). The gain comes almost entirely from putting the 28 workers to work and from breaking block 0 (the one with the long layers) into 8 pieces.

`nbody` also gained (770 to 885 GFLOP/s), which shows that the situation "grid too small for the machine" is not exclusive to one program.

`matrix-rotate` continued to be studied after this: the next step, which took the time to close to 4 ms, is in [19-automatic-prefetch.md](19-automatic-prefetch.md).

## Limitations and trade-offs

- **More calls.** Each group becomes a separate call, with its own `claim` and its own `blockIdx` computation. For large grids, this would be cost without benefit; hence the limit of 8 blocks per worker.
- **Imbalance within the group remains.** Splitting into groups of 32 lanes solves the imbalance between blocks, but not within a group: in `matrix-rotate`, the group that contains layer 0 is still longer than the others. With 80 tasks, however, that group is only a small fraction of the total.
- **Fixed limit.** The number 8 (blocks per worker) is a simple rule, not a per-kernel measurement. A kernel with, say, 300 very uneven blocks would not be split.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): blocks, warps, lanes and workers.
- [04-block-order.md](../iteration-1/04-block-order.md): how the unit number becomes `blockIdx`.
- [15-smt-cooperative-variant.md](15-smt-cooperative-variant.md): the case in which a block is run by two CPUs at the same time.
- [17-block-distribution.md](17-block-distribution.md): how units are handed out to workers.
- [19-automatic-prefetch.md](19-automatic-prefetch.md): the continuation of the `matrix-rotate` story.
- [22-what-did-not-work.md](22-what-did-not-work.md): discarded attempts.
