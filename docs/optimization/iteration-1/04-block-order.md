# 04. Block order in bands (thread-block swizzle)

> **In one sentence:** instead of handing the grid's blocks to the workers in row-major order, the executor hands them out in vertical bands of 8 columns, so that the blocks running at the same time share their working set and it stays in L3.

## The problem

A CUDA kernel is launched over a **grid** of blocks (see [00-basic-concepts.md](../00-basic-concepts.md)). In a matrix multiplication `C = A x B`, each block computes one tile of `C`. To do so it reads a **horizontal strip of A** (the rows of the tile, with all the columns of A) and a **vertical strip of B** (the columns of the tile, with all the rows of B).

The executor has 28 workers (OS threads pinned by affinity to logical CPUs 2-15 and 18-31). Each worker claims the next free block index, executes the block and repeats. Before this optimization, the linear index was translated directly into row-major order: block 0 is `(x=0, y=0)`, block 1 is `(x=1, y=0)`, and so on, finishing an entire grid row before starting the next one.

With 4096 x 4096 `float` matrices, each matrix occupies 67 MB. The Ryzen 9 8940HX has two CCDs, each with 32 MB of L3 shared only by the 8 cores of that CCD (one CCD's L3 does not serve lines to the other; a miss goes to DRAM or, for modified data, to the other CCD over the Infinity Fabric). Therefore **the whole of B does not fit in any L3**, not even summing the two.

At any instant, the 28 workers are running 28 consecutive blocks of the same grid row:

```text
Grid of blocks (each letter is a block; "#" = running now)

         grid columns ->
row 0:   # # # # # # # # # # # # # # # # # # # # # # # # # # # # . . . . . . . .
row 1:   . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
row 2:   . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
```

These 28 blocks share **the same strip of A** (they are all in the same row), which is great. But each one reads **a different strip of B**. As the workers advance along row 0, they end up reading the whole of B (67 MB). When they reach the end of row 0 and start row 1, the first columns of B were evicted from the cache long ago, because B does not fit in it. Since Zen 4's L3 is a victim cache with approximately LRU replacement, a cyclic sweep over a set larger than the capacity is the worst case: each line is evicted before being reused. Result: **B is re-read from DRAM on every grid row**.

## The idea

Reuse depends on which blocks are executing simultaneously. In row-major order, the 28 active blocks form a 28 x 1 region of the grid: they reuse one strip of A and touch 28 distinct strips of B. For the same number of blocks, a region closer to a square minimizes the sum "strips of A + strips of B" touched and, therefore, the active working set.

On GPUs the problem is the same (with L2 in place of L3), and the classic solution is the **thread-block swizzle**: remap the linear block index to other coordinates. Instead of sweeping the grid one whole row at a time, we divide the grid into **vertical bands of 8 columns** and sweep each band row by row before moving to the next band:

```text
Linear order (before)             Band order (after), kBand = 8

 0  1  2  3 ...  15  16 ...        0  1  2  3  4  5  6  7 | 32 33 ...
 .  .  .  .       .   .            8  9 10 11 12 13 14 15 | 40 41 ...
                                  16 17 18 19 20 21 22 23 | ...
                                  24 25 26 27 28 29 30 31 | ...
```

With this, the 28 active blocks occupy a region of approximately **8 columns x 3.5 rows** of the grid. They touch only 8 strips of B and 3 or 4 strips of A. While the band of columns is traversed from top to bottom, the same 8 strips of B are reused by all the blocks of that band and stay in L3.

## How it works, step by step

The executor does not change the block counter. The workers keep taking sequential numbers: 0, 1, 2, 3... What changes is the **translation** of that linear number `b` into the `(x, y, z)` coordinates of the block, which is what the kernel sees as `blockIdx`.

Example with a small grid: `gx = 10` columns, `gy = 3` rows, `kBand = 8`.

1. Each complete band has `kBand * gy = 8 * 3 = 24` blocks.
2. For a number `b`, the band is `b / 24`. The first column of the band is `band * 8`.
3. The last band can be narrower: here band 1 starts at column 8 and has only `10 - 8 = 2` columns.
4. Within the band, the position is `in_band = b - band * 24`. The column is `start + in_band % width` and the row is `in_band / width`.

Result for all 30 blocks:

```text
             x=0  x=1  x=2  x=3  x=4  x=5  x=6  x=7 | x=8  x=9
row y=0:       0    1    2    3    4    5    6    7 |  24   25
row y=1:       8    9   10   11   12   13   14   15 |  26   27
row y=2:      16   17   18   19   20   21   22   23 |  28   29
```

Each block appears exactly once: the translation is a **permutation** of `[0, gx*gy)`.

Traffic estimate for the 4096 x 4096 multiplication with 16 x 16 blocks (a grid of 256 x 256 blocks); this is a rough calculation, not a measurement:

- **Linear order:** each grid row reads the whole of B. There are 256 rows, so B (67 MB) comes from DRAM about 256 times.
- **Band order:** the 8 strips of B of a band of columns (8 x 16 columns x 4096 rows x 4 bytes = 2 MB) fit comfortably in the cache and are read once per band. Each band of columns has to re-read the whole of A, but there are only 256 / 8 = 32 bands.

The trade is: we re-read A a few dozen times instead of re-reading B hundreds of times.

### Why this is allowed

The CUDA programming model **guarantees no order** of execution among blocks. A correct CUDA program cannot depend on block 0 running before block 1. For this reason the executor can choose whatever order it wants, without changing the result of any correct program.

## The code

The translation lives in `Executor::block_coords`.

`src/device/executor.cpp:198-225`

```cpp
void Executor::block_coords(const Job& job, uint64_t b, uint32_t* idx) const {
    const uint32_t gx = job.cfg.grid[0], gy = job.cfg.grid[1];
    // Block order (CUDA promises none): for 2D grids, bands of kBand columns
    // walked row by row, so the blocks running at the same time touch a few
    // rows and a few columns of the data instead of one row and many columns
    // (the classic thread-block swizzle; it keeps both operands of a matmul in
    // cache). CPUSIM_BLOCK_ORDER=linear disables it.
    static const bool linear = [] {
        const char* v = std::getenv("CPUSIM_BLOCK_ORDER");
        return v && std::strcmp(v, "linear") == 0;
    }();
    constexpr uint32_t kBand = 8;
    const bool swizzle = !linear && gy > 1 && gx > kBand;
    uint64_t plane = uint64_t(gx) * gy;
    idx[2] = uint32_t(b / plane);
    uint64_t r = b % plane;
    if (swizzle) {
        uint64_t band = r / (uint64_t(kBand) * gy);       // which band of columns
        uint32_t band_x0 = uint32_t(band * kBand);
        uint32_t width = std::min(kBand, gx - band_x0);    // the last band may be narrower
        uint64_t in_band = r - band * uint64_t(kBand) * gy;
        idx[0] = band_x0 + uint32_t(in_band % width);
        idx[1] = uint32_t(in_band / width);
    } else {
        idx[0] = uint32_t(r % gx);
        idx[1] = uint32_t(r / gx);
    }
}
```

What each part does:

- `linear` reads the environment variable `CPUSIM_BLOCK_ORDER` only once (initialization of a local `static`, thread-safe since C++11). If it equals `linear`, the optimization is turned off. This serves to compare the two orders.
- `swizzle` is only turned on when it makes sense: the grid must have more than one row (`gy > 1`) and more than 8 columns (`gx > kBand`). On a grid with a single row, or with up to 8 columns, the remapping would be the identity or would not change the working set.
- `plane` is the number of blocks in a `z` "slice". The `z` coordinate (`idx[2]`) stays in simple order; the bands are made within each slice.
- The `if (swizzle)` branch implements the calculation of the example above: it finds the band (`band`), its first column (`band_x0`), the width (8, or less in the last band) and the position within the band (`in_band`).
- The `else` branch is the old linear order: the column is the remainder of the division by `gx`, the row is the quotient.

The caller of this function is each worker's loop, right after claiming the next block number:

`src/device/executor.cpp:144-147`

```cpp
    uint64_t b;
    while (claim(job, worker, &b)) {
        uint32_t idx[3];
        block_coords(job, b, idx);
```

The worker takes a number `b` with `claim` and only then converts it to coordinates. The same function is also used in the mode where a block is split into groups of warps and in the cooperative mode with pairs of workers (see [15-smt-cooperative-variant.md](../iteration-2/15-smt-cooperative-variant.md) and [16-splitting-blocks-into-warps.md](../iteration-2/16-splitting-blocks-into-warps.md)), so all modes inherit the band order.

A historical note: in the old record (`OPTIMIZATIONS.md`) this logic appears inside `run_blocks`. Today it has been isolated in `block_coords`, and the way block numbers reach the workers (`claim`) also changed: it stopped being a single shared counter and became a distribution of its own, described in [17-block-distribution.md](../iteration-2/17-block-distribution.md). The band mapping, however, remains the same: whatever number the worker receives, it is translated by this function.

## When it applies (and when it does not)

- **It applies** to 2D grids (or 3D, slice by slice) with more than one row and more than 8 columns. This is the typical case of matrix multiplication, transposition, 2D stencils and image processing.
- **It does not apply** to 1D grids (`gy == 1`), which are the majority of simple kernels (adding vectors, for example). In those, consecutive blocks already touch neighboring data, so the linear order is already the good one.
- **It does not apply** to grids with 8 columns or fewer: each row is already a band.
- **It can be turned off** with `CPUSIM_BLOCK_ORDER=linear`.
- **Correctness:** it never changes the result of a correct CUDA program, because CUDA promises no order among blocks. A program that (incorrectly) depends on that order could behave differently, but it would also behave unpredictably on a real GPU.
- **It may not help** when the blocks of a column do not share data (for example, a kernel in which each block reads only its own region of the input). In that case the order does not matter for the cache and the cost of the extra computation is negligible (a few integer divisions per block).

## Results

Measured on the matrix multiplication benchmarks (`bench/`), when the optimization landed:

| Kernel | Change in time |
|---|---|
| matmul naive (no shared memory) | -27% |
| matmul tiled (with shared memory) | -28% |

The gain comes almost entirely from fewer DRAM reads: matrix B stops being re-read on every grid row.

## Limitations and trade-offs

- The band width is fixed (`kBand = 8`). It was chosen so that the 28 simultaneous blocks form a region close to a square (8 x 3.5). With another number of workers or another cache size, another value might be better; there is no automatic tuning.
- The band order trades re-reads of B for re-reads of A. If A were much larger than B, the trade could get worse; in the square multiplication this does not happen.
- The executor does not look at the kernel to decide: it applies the same order to any 2D grid large enough.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): grid, block, cache, DRAM.
- [17-block-distribution.md](../iteration-2/17-block-distribution.md): how block numbers reach the workers today.
- [15-smt-cooperative-variant.md](../iteration-2/15-smt-cooperative-variant.md) and [16-splitting-blocks-into-warps.md](../iteration-2/16-splitting-blocks-into-warps.md): other modes that also use `block_coords`.
- [19-automatic-prefetch.md](../iteration-2/19-automatic-prefetch.md): another way to hide memory slowness.
