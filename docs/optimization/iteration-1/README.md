# Iteration 1

The first round of optimizations: it made it viable to run CUDA on the CPU with reasonable performance. The starting point was kernels 2 to 25 times slower than at the end of this iteration (matmul naive from 2400 to 224 ms, tiled from 3179 to 304 ms).

The record written at the time is in [OPTIMIZATIONS.md](OPTIMIZATIONS.md), kept as it was written, with the table of overall results and the measurement tools. The documents below explain each optimization in detail.

| # | Document | In one line |
|---|---|---|
| 01 | [Specialized JIT](01-specialized-jit.md) | Recompiling the kernel at launch, knowing the block size |
| 02 | [Width variants](02-width-variants.md) | Executing 32, 64, 128... threads per call, with and without suspending at barriers |
| 04 | [Block order](04-block-order.md) | Walking the grid in bands so that simultaneous blocks share cache |
| 05 | [Address decomposition](05-address-decomposition.md) | Separating the common part of the address from the part that changes per thread |
| 06 | [Dynamic coalescing](06-dynamic-coalescing.md) | Replacing gathers with contiguous loads when the pattern allows |
| 07 | [Uniform loops](07-uniform-loops.md) | Loops that all threads execute identically do not need masks |
| 08 | [If-conversion](08-if-conversion.md) | Turning small `if`s into selects, without a branch |
| 09 | [State at barriers](09-state-at-barriers.md) | Saving less state when a warp stops at a `__syncthreads` |
| 11 | [LLVM tuning](11-llvm-tuning.md) | Extra optimizer options and passes |

The first version of the automatic width selection is also from this iteration; it was redone in iteration 2 and is described in [03](../iteration-2/03-automatic-width-selection.md). Documents 06, 09 and 11 received additions in iteration 2, indicated at the top of each one.

Next: [iteration 2](../iteration-2/README.md). Back to the [index](../README.md).
