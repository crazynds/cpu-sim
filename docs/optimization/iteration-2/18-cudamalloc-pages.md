# 18. Memory pages reserved in cudaMalloc

> **In one sentence:** when the program calls `cudaMalloc`, the simulator now makes the Linux kernel materialize (allocate, zero and map) all the pages of that memory before returning, as a GPU does; this way the first kernel that writes to a new vector is not interrupted midway by hundreds of 2 MiB page faults.

The project's general concepts (simulated VRAM, worker) are in [00-basic-concepts.md](../00-basic-concepts.md). The Linux mechanism involved:

- **THP (Transparent Huge Pages)**: the simulated VRAM is an anonymous region marked with `madvise(MADV_HUGEPAGE)`. On a first-touch fault in a 2 MiB-aligned stretch entirely contained in the VMA, the kernel (`do_huge_pmd_anonymous_page`) tries to allocate an order-9 compound page (2 MiB contiguous), zeroes it entirely and maps it with a single PMD entry, with no PTE-level table. If there are no free contiguous 2 MiB, it falls back to 4 KiB pages (`khugepaged` may collapse them later). The gain is in TLB reach: on Zen 4 the L2 DTLB has 3072 entries, which covers 12 MiB with 4 KiB pages and 6 GiB with 2 MiB pages; for kernels that sweep hundreds of MB this eliminates almost all page walks.
- **Cost of the first-touch fault with THP**: the fault is synchronous in the thread that touched the address, and includes allocating the compound page and zeroing 2 MiB (512 times the work of a 4 KiB fault, concentrated in a single event).

## The problem

### Address reservation versus resident memory

An anonymous `mmap` only creates the VMA; no physical page is allocated until the first access (demand paging). The simulated VRAM is a large VMA of this kind, and a `cudaMalloc` just picks a free stretch of it and returns the address; no page is materialized.

What then happens in the first kernel that writes to a newly allocated vector:

```text
worker 3 writes to out[0]          -> page fault: OS zeroes 2 MiB  (~100 us or more)
worker 3 writes to out[1]          -> ok, the page already exists
...
worker 7 writes to out[524288]     -> page fault: OS zeroes another 2 MiB
...
```

Each new page costs about **100 microseconds or more** (zeroing 2 MiB and mapping them). A 400 MB vector has 200 pages of 2 MiB. This cost shows up **in the middle of the kernel's execution**, spread unpredictably among the workers.

On a GPU this does not happen: when `cudaMalloc` returns, the memory already physically exists on the card.

### The side effect on width selection

The problem was not only one of time. The simulator automatically chooses the best "width" for each kernel (how many lanes per call) by measuring slices of the first launch with each candidate ([03-automatic-width-selection.md](03-automatic-width-selection.md)). If a slice coincided with the moment when the pages were being created, it looked slow, and the simulator could choose the wrong width.

In `black-scholes`, the same measurement gave very different results:

| Situation              | Variation among measured slices |
| ---------------------- | ------------------------------- |
| Without pre-allocation | from 8.7 to 14.2                |
| With VRAM pre-allocated | from 7.0 to 7.4                |

With the variation from 8.7 to 14.2, the "measurement" was measuring the operating system, not the kernel.

## The idea

Since the cost of the fault is unavoidable, the idea is to bring it forward into `cudaMalloc`, which is where a GPU pays for the allocation: on return, all the pages of the stretch are already resident and mapped.

Linux (from 5.14) has an operation for exactly this: `madvise(addr, len, MADV_POPULATE_WRITE)`. It walks the range and executes, inside the kernel and in a single syscall, the same path as a write fault for each page not yet mapped (allocating, zeroing and mapping with write permission, and breaking COW if any), without writing anything to the contents. Unlike `MAP_POPULATE`, which only applies to `mmap` and to the whole region, it can be applied to arbitrary subranges, which is what an allocator needs. In a VMA with `MADV_HUGEPAGE`, the aligned 2 MiB stretches are populated as huge pages by the same path as the fault. On kernels without this operation, `madvise` returns `EINVAL`, and the alternative is to write one byte to each page from user space, which causes the same faults now instead of during the launch, at the cost of one kernel entry per page.

## How it works, step by step

Reduced example: 2 MiB pages, and a 5 MiB `cudaMalloc` that landed at offset 3 MiB of the VRAM.

```text
VRAM:   |   page 0   |   page 1   |   page 2   |   page 3   |  page 4  |
        0          2 MiB        4 MiB        6 MiB        8 MiB
allocation:            [3 MiB ........................ 8 MiB)
```

1. The allocator picks the stretch [3 MiB, 8 MiB).
2. `populate` rounds to whole pages: the start goes down to the beginning of the page (2 MiB) and the end goes up to the end of the page (8 MiB).
3. It calls `madvise(base + 2 MiB, 6 MiB, MADV_POPULATE_WRITE)`. The OS creates pages 1, 2 and 3.
4. If `madvise` fails (old kernel), it writes a zero to the first byte of pages 1, 2 and 3.
5. `cudaMalloc` returns. The kernel that comes afterwards finds no page fault in this vector.

Page 1 is also shared with whatever lies between 2 MiB and 3 MiB; if it already existed, there is no extra cost.

When the memory is freed (`cudaFree`), the whole pages of the stretch are returned to the system (`MADV_DONTNEED`), so as not to hold the computer's RAM. A later `cudaMalloc` at the same place creates them again.

## The code

### The declaration

`src/device/vram.h:34-37`

```cpp
    // Backs [begin, end) (offsets) with RAM now, as a GPU's cudaMalloc does:
    // kernels never stop on first-touch page faults (with 2 MiB pages, each
    // costs a 2 MiB clear in the middle of a launch).
    void populate(uint64_t begin, uint64_t end);
```

The comment summarizes the reason: with 2 MiB pages, each page fault costs zeroing 2 MiB in the middle of a launch.

### The implementation

`src/device/vram.cpp:62-71`

```cpp
void VramPool::populate(uint64_t begin, uint64_t end) {
    if (prefault_ || end <= begin) return;
    uint64_t b = begin / page_size_ * page_size_;
    uint64_t e = std::min(size_, (end + page_size_ - 1) / page_size_ * page_size_);
#ifdef MADV_POPULATE_WRITE
    if (madvise(base_ + b, e - b, MADV_POPULATE_WRITE) == 0) return;
#endif
    // Older kernels: touch every page.
    for (uint64_t off = b; off < e; off += page_size_) *reinterpret_cast<volatile uint8_t*>(base_ + off) = 0;
}
```

- If the whole VRAM was already pre-created at initialization (`prefault_`, a configuration option), there is nothing to do.
- `b` and `e` are the start and the end rounded to whole pages, as in the example.
- `#ifdef MADV_POPULATE_WRITE`: it only tries the call if the system on which the simulator was compiled knows it.
- The fallback loop writes one byte per page; the `volatile` prevents the write from being eliminated as a dead store.

### Where it is called

`src/device/device_core.cpp:194-215`

```cpp
Error DeviceCore::allocate_in(RangeAllocator& pool, uint8_t* pool_base, uint32_t id, uint64_t bytes, uint64_t* dptr) {
    counters_.alloc_calls.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex_);
    // ...
    auto off = pool.allocate(bytes);
    if (!off) {
        counters_.failed_allocs.fetch_add(1, std::memory_order_relaxed);
        return Error::MemoryAllocation;
    }
    auto [start, size] = *pool.find(*off, 0);
    *dptr = reinterpret_cast<uint64_t>(pool_base) + start;
    if (&pool == allocator_.get()) vram_->populate(start, start + size);
    it->second.allocations.emplace(*dptr, size);
    // ...
    return Error::Success;
}
```

`allocate_in` serves both the ordinary `cudaMalloc` (VRAM) and managed memory (`cudaMallocManaged`, which uses another allocator). The line `if (&pool == allocator_.get())` guarantees that only the **VRAM** is populated. Managed memory is also accessed by the program on the host side, and there the normal behavior (creating the page on first touch) is what is expected.

### The 2 MiB pages

`src/device/vram.cpp:45-47`

```cpp
    // Transparent huge pages: needs "madvise" or "always" in
    // /sys/kernel/mm/transparent_hugepage/enabled; silently 4 KiB pages otherwise.
    if (hugepages == HugePages::TwoMiB) madvise(p, pool->size_, MADV_HUGEPAGE);
```

This is where the VRAM marks the VMA as eligible for THP. With `enabled=never`, `MADV_HUGEPAGE` is ignored and the pages stay at 4 KiB; the `populate` via `MADV_POPULATE_WRITE` remains complete, just with 512 times more pages. Note that `page_size_` reflects the requested size (`vram.cpp:23`), not the obtained one: if the kernel falls back to 4 KiB, the rounding to 2 MiB remains valid for `madvise`, but the fallback loop, which advances by `page_size_` at a time, would touch only the first 4 KiB page of each 2 MiB stretch.

### The return in cudaFree

`src/device/vram.cpp:55-60`

```cpp
void VramPool::discard(uint64_t begin, uint64_t end) {
    if (prefault_) return;
    uint64_t b = (begin + page_size_ - 1) / page_size_ * page_size_;
    uint64_t e = end / page_size_ * page_size_;
    if (e > b) madvise(base_ + b, e - b, MADV_DONTNEED);
}
```

The rounding here is the opposite of `populate`: it only returns pages that are **entirely** inside the freed stretch, so as not to erase data of a neighboring allocation that shares the same page.

## When it applies (and when it does not)

- **Always**, in every VRAM `cudaMalloc`, except when the whole VRAM was pre-created at initialization (`prefault` option).
- It does **not** apply to managed memory (`cudaMallocManaged`).
- The gain shows up in the **first** kernel that touches each new vector. Programs that allocate once and run many launches already paid the cost only once; what changes for them is mainly the **stability of the width measurement**.
- Programs that allocate and free vectors repeatedly inside a loop pay the cost at each `cudaMalloc` (before, they paid in the following kernel; the total is similar).

## Results

The measured number is in `black-scholes`: the variation among the slices of the width measurement dropped from **8.7 to 14.2** (without pre-allocation) to **7.0 to 7.4** (with the VRAM pre-allocated). With stable slices, the automatic width choice comes to reflect the kernel, not the operating system.

There was no isolated measurement of the total time of each program with only this change.

## Limitations and trade-offs

- **The cost moves, it does not disappear.** Zeroing the pages still costs the same; now it happens inside `cudaMalloc`. For those who measure only the kernel (as the benchmarks and our width selection do), it is better. For those who measure the whole program, it is the same.
- **Pays for what it does not use.** A program that allocates 2 GB "just in case" and uses only 100 MB now actually occupies the 2 GB of RAM. This is exactly what happens on a real GPU (the allocated VRAM is reserved, used or not), so the behavior becomes more faithful, but it consumes more of the computer's RAM than before.
- **Depends on the Linux kernel.** `MADV_POPULATE_WRITE` exists from Linux 5.14. On older systems, the fallback touches page by page, which is slower; if THP has fallen back to 4 KiB, it pre-allocates only part of the pages (see above), and the rest goes back to being allocated on first touch during the kernel.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): VRAM, kernel, worker.
- [03-automatic-width-selection.md](03-automatic-width-selection.md): the slice-based measurement that was becoming unstable.
- [17-block-distribution.md](17-block-distribution.md): another case in which memory (cache, in this case) decided performance.
- [22-what-did-not-work.md](22-what-did-not-work.md): other measurement pitfalls.
