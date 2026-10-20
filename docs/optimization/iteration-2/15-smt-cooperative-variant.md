# 15. Cooperative variant: the two SMT threads of a core sharing a block

> **In one sentence:** in the cooperative variant (suffix `c`, for example `128c`), the two logical (SMT) threads of one physical core execute the two halves of a CUDA block **at the same time**, sharing the same shared memory, and each `__syncthreads` becomes a real, very short wait between the two, generated directly in the code, without saving state and without suspending anyone.

> **Since then:** removed in iteration 4. Measured per kernel over 47 HeCBench programs, it never won and was 2 to 50 times slower than the best width: [36](../iteration-4/36-cooperative-variant-false-sharing-and-vram.md).

The project's general concepts (worker, shared memory, variants) are in [00-basic-concepts.md](../00-basic-concepts.md). The handling of the barrier in the other variants is in [09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md).

## The problem

The executor uses 14 Zen 4 physical cores, each with two hardware threads (**SMT**, *simultaneous multithreading*; Hyper-Threading in Intel's nomenclature). The two logical threads of a core, called **siblings** here, have their own architectural state (registers, including the 32 zmm), but share the pipeline: schedulers, integer and floating-point/vector execution units, the 32 KiB L1D and the 1 MiB L2; some internal queues are statically partitioned between the two. The executor has 28 workers, one pinned to each logical thread.

A kernel with `__syncthreads` has, so far, two ways of running a block on a worker:

1. **Suspending variants** (32 lanes, `64s`, `128s`): each group of lanes runs up to the barrier, writes the live state to the frame, returns, and is called again later ([09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md)). Each barrier costs context switches.
2. **Whole-block variants** (64 to 512 lanes): the whole block in a single call, so the barrier costs nothing. But, with 256 or 512 lanes, each varying `float`/`i32` value occupies 16 or 32 zmm registers. With 32 architectural zmm, two live values of 256 lanes already exhaust the register file, and the allocator does **spill**: each saved zmm is a 64-byte `vmovups` to the stack and a later reload, which compete with the kernel's own loads and stores for the memory ports.

The question that motivated this variant (an idea from the user): what if the block were split in half between the two sibling threads of a core? Each would have half of the lanes (half the register pressure, and each sibling has its own file of 32 zmm), both would access the same shared memory through the same L1D, and the barrier would reduce to a synchronization between two threads, without suspending.

## The idea

With exactly two participants, a barrier needs neither an atomic read-modify-write operation (`lock xadd`, `lock cmpxchg`) nor a shared counter. A monotonic counter per participant is enough, the **epoch**, each with a single writer: on arriving at the barrier, the half increments its own epoch and waits until the other's epoch is at least equal. Since the two halves execute the same number of barriers per block, "the other already has epoch `>= e`" means "the other has already reached barrier `e`".

The memory ordering comes from the release/acquire pair: the *release* write of the epoch makes visible, before it, all earlier writes of this half (including those to shared memory), and the *acquire* read that observes the sibling's epoch guarantees that the sibling's writes preceding the barrier are visible after it. A full barrier (`mfence`) between the own write and the read of the sibling's epoch is not needed: if the store stays held in the store buffer while the other half reads the old value, the effect is only that it spins a few more times in the wait loop; correctness does not depend on store-load ordering, unlike a mutual exclusion protocol such as Dekker's.

This turns each `__syncthreads` into a few inline instructions, without saving state and without returning to the executor.

## How it works, step by step

### Who runs what

For a block of `L+1` up to `2L` threads, the cooperative variant of `L` lanes is used. Example: a block of 256 threads with the `128c` variant:

```text
physical core 3
  logical thread A (worker "half 0"): lanes   0..127 of the block  (warp_index 0)
  logical thread B (worker "half 1"): lanes 128..255 of the block  (warp_index 4)
  shared memory: a single buffer, worker A's, used by both
```

The compiler generates cooperative variants of 64, 128, 256 and 512 lanes for every kernel with a barrier. The choice of which variant to use is made by measurement ([03-automatic-width-selection.md](03-automatic-width-selection.md)); it can also be forced with `CPUSIM_LANES=128c`.

### The sibling pairs come from the system

Which logical threads are siblings is not deduced from the CPU numbering (which varies from machine to machine): the executor reads the topology from `/sys` and forms a `Pair` for each core whose two threads are both among the workers. Workers without a sibling among the workers do not take part in cooperative launches.

### The epoch barrier

Each half has a 64-bit counter, the **epoch**, on its own cache line (see the discussion in "The pairs"). It is divided into two parts:

```text
epoch = (round << 32) | barriers
          high 32 bits   low 32 bits
```

- **round**: how many blocks the pair has already started (1, 2, 3, ...);
- **barriers**: how many barriers of this half have already been crossed in this block.

The rules:

1. On **starting** the block of round `r`, the half writes `r << 32` (barriers = 0).
2. At each `__syncthreads`, the half computes `e = epoch + 1`, writes `e`, and waits until the sibling's epoch is `>= e`.
3. On **finishing** the block, the half writes `(r << 32) | 0xffffffff`: the largest possible value for that round. This way the sibling, if it still has barriers ahead, never waits for it again.

Example: round 5, kernel with 2 barriers:

```text
                       epoch A              epoch B              comment
B still in round 4     5<<32 | 0           4<<32 | ffffffff      B finished the previous block
A reaches barrier 1    5<<32 | 1           4<<32 | ffffffff      A waits: (5<<32)-1 < (5<<32)+1
B starts round 5       5<<32 | 1           5<<32 | 0
B reaches barrier 1    5<<32 | 1           5<<32 | 1             both see >=, both proceed
A reaches barrier 2    5<<32 | 2           5<<32 | 1             A waits
B reaches barrier 2    5<<32 | 2           5<<32 | 2             both proceed
A finishes             5<<32 | ffffffff    5<<32 | 2
B finishes             5<<32 | ffffffff    5<<32 | ffffffff
```

Note the first row: `4<<32 | 0xffffffff` is exactly `(5<<32) - 1`. So "the sibling finished the previous round" is automatically **smaller** than any barrier of the new round, and A waits for B to start. A single comparison covers both cases. The counters start at `0xffffffff` (round 0 finished).

The barrier is only counted if some lane of that half is active at that point in the code. A half whose lanes have all left a loop (divergence) no longer executes the barriers of that loop; counting barriers only with active lanes keeps the two counts equal when the CUDA program is valid, that is, when every thread of the block executes the same `__syncthreads`.

### Why inline, without calling a function

The first version called a runtime function to perform the barrier. In the System V x86-64 calling convention (Linux), **all** vector registers (`xmm`/`ymm`/`zmm` 0 to 31) and the mask registers `k0` to `k7` are *caller-saved*: the called function may destroy them without restoring. (In the Windows x64 convention, only the low 128 bits of `xmm6` to `xmm15` are preserved, which would not help with zmm either.) Since LLVM does not know the body of the runtime function, it has to assume the worst: before each call, it saves on the stack every vector live across it and reloads it afterwards. In a kernel with 128 lanes per half, each live value is 8 zmm. Count of zmm accesses to the stack in tiled matmul:

| version | zmm accesses to the stack |
|---|---|
| `128c` with function call | 833 |
| `128c` with inline barrier | 331 |
| `128c` inline, after the static analysis fixes | 15 |
| `256` (whole block), same point in time, for comparison | 71 |

The final version generates the barrier as a small piece of code inside the kernel itself: a read, an add, a write and a wait loop with the `pause` instruction. `pause` is the x86 *spin-wait* hint: it delays the execution of the next instruction, which reduces the rate of speculative loads in the loop, avoids the pipeline flush caused by a memory-ordering violation when leaving the loop, and yields the core's shared resources to the SMT sibling, which is precisely what the loop is waiting for. Since the code no longer contains any call, the register allocator keeps the vectors in zmm across the barrier. The "static analysis fixes" are later compiler improvements (see, for example, [12-static-access-analysis.md](12-static-access-analysis.md)) that reduced the code and the live values; the comparison with the 256-lane variant was made at the same point in time.

### Handing blocks over within the pair

The worker "half 0" obtains the next block from an atomic counter of the launch (`fetch_add`) and publishes it to the sibling in one of two slots (`slot[r & 1]`), together with the shared memory pointer. Before obtaining block `r`, it waits for the sibling to have finished block `r - 1` (its epoch `>= ((r-1) << 32) | 0xffffffff`): only then can the shared memory be reused. The worker "half 1" waits for the slot to announce round `r` (acquire load of `round`, which pairs with half 0's release store) and then reads `block` and `shared`, non-atomic fields whose visibility is guaranteed by this pair.

### Cooperative launches run alone

Every sibling pair must take part, and each paired worker enters the launch **exactly once** (the pair counts rounds together). For this reason a cooperative launch waits for the executor to become empty, runs **exclusively**, and only then do other launches enter again. This turns off, for it, the overlap between kernels of [20-concurrent-kernels.md](20-concurrent-kernels.md).

## The code

### The contract

`src/device/kernel_abi.h:42-48`

```cpp
// And "cooperative" variants of 64, 128, 256 and 512 lanes,
//     uint32_t __cpusim_entry<L>c_<mangled name>(const cpusim_warp_ctx*);
// for blocks of L+1 to 2L threads: the two SMT siblings of a core run the two
// halves of one block at the same time (warp_index 0 and L/32), sharing its
// shared memory. Each __syncthreads is inline code (no call: a call would
// spill every vector register): bump this half's epoch, wait until the other
// half's epoch is at least as high. No frames, no suspending.
```

`src/device/kernel_abi.h:75-80`

```cpp
    // Cooperative variants: barrier epochs of this half and of the other one,
    // each on its own cache line. The runtime sets both halves' epochs to the
    // same value before a block, and a half that returns sets its epoch to
    // that value | 0xffffffff, so the other one never waits for it again.
    uint64_t* coop_epoch;
    const uint64_t* coop_other;
```

`src/device/kernel_abi.h:153-154`

```cpp
#define CPUSIM_CTX_COOP_EPOCH 80
#define CPUSIM_CTX_COOP_OTHER 88
```

Two new pointers in the context header, at offsets 80 and 88: this half's epoch (which it writes) and the sibling's (which it only reads, hence `const`). These fields took the contract to version 8 (`CPUSIM_KERNEL_ABI 8`, line 26).

### The generated barrier

`src/compiler/simt.cpp:1816-1852`

```cpp
// Cooperative variant: wait for the other half of the block, inline.
//     e = *epoch + 1; atomic store-release *epoch = e;
//     while (atomic load-acquire *other < e) pause;
// Only when some lane of this half is here, so the two halves count the same
// barriers (a half that already returned stops counting, see kernel_abi.h).
void WarpBuilder::emit_coop_barrier(Value* mask) {
    auto* ptr = PointerType::get(ctx_, 0);
    auto* i64 = b_.getInt64Ty();
    BasicBlock* arrive = BasicBlock::Create(ctx_, "coop.arrive", w_);
    BasicBlock* wait = BasicBlock::Create(ctx_, "coop.wait", w_);
    BasicBlock* spin = BasicBlock::Create(ctx_, "coop.spin", w_);
    BasicBlock* after = BasicBlock::Create(ctx_, "coop.after", w_);
    b_.CreateCondBr(any_lane(mask), arrive, after);

    b_.SetInsertPoint(arrive);
    // ...
    Value* epoch = field(CPUSIM_CTX_COOP_EPOCH, "coop.epoch");
    Value* other = field(CPUSIM_CTX_COOP_OTHER, "coop.other");
    LoadInst* cur = b_.CreateAlignedLoad(i64, epoch, Align(8), "coop.cur");
    cur->setAtomic(AtomicOrdering::Monotonic);
    Value* next = b_.CreateAdd(cur, ConstantInt::get(i64, 1), "coop.next");
    b_.CreateAlignedStore(next, epoch, Align(8))->setAtomic(AtomicOrdering::Release);
    b_.CreateBr(wait);

    b_.SetInsertPoint(wait);
    LoadInst* seen = b_.CreateAlignedLoad(i64, other, Align(8), "coop.seen");
    seen->setAtomic(AtomicOrdering::Acquire);
    b_.CreateCondBr(b_.CreateICmpUGE(seen, next), after, spin);

    b_.SetInsertPoint(spin);
    b_.CreateCall(Intrinsic::getDeclaration(&m_, Intrinsic::x86_sse2_pause));
    b_.CreateBr(wait);

    b_.SetInsertPoint(after);
}
```

Four code blocks: `coop.arrive` (increments the own epoch), `coop.wait` (reads the sibling's and compares), `coop.spin` (executes `pause` and compares again) and `coop.after` (the kernel continues). The memory orderings are those described in "The idea": the **release** store of the own epoch publishes the earlier writes of this half (shared memory included), and the **acquire** load of the sibling's epoch, when it observes `>= next`, establishes *happens-before* with everything the sibling wrote before the barrier. The read of the own epoch is `Monotonic` (relaxed): only this half writes that counter during the call, so it is enough for the read to be atomic. On x86, which is TSO, the three accesses compile to plain `mov`; the orderings act on the compiler, forbidding it from moving shared memory accesses to the other side of the barrier. The only call is to the `pause` intrinsic, which becomes an instruction, not a function call.

`src/compiler/simt.cpp:2496-2499`

```cpp
        if (is_barrier(*call)) {
            if (coop_) emit_coop_barrier(mask);
            return true;
        }
```

In both the whole-block and the cooperative code, the barrier is handled when the instruction is emitted: in the whole-block variants it simply disappears; in the cooperative one it becomes the wait above. (The comment of the `WarpBuilder` constructor, at `src/compiler/simt.cpp:65-66`, describes exactly this: an inline 2-part barrier, by epochs, without suspending.)

### The pairs

`src/device/executor.h:140-153`

```cpp
    // Two SMT siblings running blocks together. Barrier epochs (see
    // kernel_abi.h) are (round << 32) | barriers passed, one cache line each;
    // round r's block goes from the first worker of the pair to the second
    // through slot[r & 1].
    struct alignas(64) Pair {
        alignas(64) std::atomic<uint64_t> epoch0{0xffffffffu};
        alignas(64) std::atomic<uint64_t> epoch1{0xffffffffu};
        struct alignas(64) Slot {
            std::atomic<uint64_t> round{0};
            uint64_t block = 0;
            uint8_t* shared = nullptr;
        } slot[2];
        std::atomic<uint64_t>& epoch(int half) { return half ? epoch1 : epoch0; }
    };
```

Each `alignas(64)` puts the field on its own 64-byte cache line. Between different cores, two epochs on the same line would cause *false sharing*: each write by one core would invalidate the other's copy through the coherence protocol (MOESI, on AMD) and the line would bounce between the two L1s. Between SMT siblings this classic effect does not occur, because both use the same L1D; the separation here is a low-cost precaution (64 bytes per field) that keeps each writer isolated on its own line.

`src/device/executor.cpp:21-37` (excerpt)

```cpp
Executor::Executor(const std::vector<int>& cpus) {
    // Pairs of SMT siblings, both among the workers (read from /sys, never
    // assumed from the numbering).
    pair_of_.assign(cpus.size(), -1);
    half_of_.assign(cpus.size(), 0);
    for (const PhysicalCore& core : read_topology().cores) {
        int found[2], n = 0;
        for (int cpu : core.cpus)
            for (size_t i = 0; i < cpus.size() && n < 2; ++i)
                if (cpus[i] == cpu) found[n++] = int(i);
        if (n < 2) continue;
        for (int h = 0; h < 2; ++h) {
            pair_of_[found[h]] = int(pairs_.size());
            half_of_[found[h]] = h;
        }
        pairs_.push_back(std::make_unique<Pair>());
    }
    // ...
```

For each physical core read from the topology, it looks among the workers for the two logical CPUs of that core. If both are there, it creates a `Pair` and records, for each worker, which pair and which half (0 or 1) it is.

### One worker of the pair

`src/device/executor.cpp:237-273`

```cpp
void Executor::run_pair(const Job& job, CtxArray& ctx, uint8_t* shared, uint64_t* counters, Pair& pair, int half,
                        uint64_t& round) {
    ctx = *job.warps;
    cpusim_warp_ctx* c = ctx.at(size_t(half));
    c->params = job.params;
    c->counters = counters;
    c->frame = nullptr;
    c->coop_epoch = reinterpret_cast<uint64_t*>(&pair.epoch(half));
    c->coop_other = reinterpret_cast<const uint64_t*>(&pair.epoch(1 - half));
    std::atomic<uint64_t>& mine = pair.epoch(half);
    std::atomic<uint64_t>& other = pair.epoch(1 - half);
    while (true) {
        const uint64_t r = ++round;
        Pair::Slot& slot = pair.slot[r & 1];
        uint64_t b;
        if (half == 0) {
            // The other half left block r - 1 (and the slot's last reader is done).
            while (other.load(std::memory_order_acquire) < ((r - 1) << 32 | 0xffffffffu)) _mm_pause();
            b = job.next.fetch_add(1, std::memory_order_relaxed);
            slot.block = b;
            slot.shared = shared;
            slot.round.store(r, std::memory_order_release);
        } else {
            while (slot.round.load(std::memory_order_acquire) != r) _mm_pause();
            b = slot.block;
        }
        if (b >= job.blocks) {
            mine.store(r << 32 | 0xffffffffu, std::memory_order_release);
            return;
        }
        c->shared = slot.shared;
        block_coords(job, b, c->block_idx);
        mine.store(r << 32, std::memory_order_release);
        job.entry(c);
        mine.store(r << 32 | 0xffffffffu, std::memory_order_release);
    }
}
```

Each half uses the context of index `half` (0 or 1), without a frame. At each round: half 0 waits for the sibling to release the previous block, obtains the next block and publishes it in the slot; half 1 waits for the publication. If the blocks have run out, the half marks the round as finished and leaves. Otherwise, it writes `r << 32` (start of the round), calls the generated code once for the whole block (all barriers happen inside it) and, on return, writes `r << 32 | 0xffffffff`. Note that `shared` comes from the slot: half 1 uses half 0's buffer.

### Exclusivity

`src/device/executor.cpp:275-286`

```cpp
Executor::Job* Executor::pick(int worker) {
    for (Job* j : active_) {
        if (j->cooperative) {
            // Every worker with a sibling takes part exactly once (the pair
            // hands blocks over and counts rounds together).
            if (pair_of_[size_t(worker)] >= 0 && !j->joined[size_t(worker)]) return j;
            continue;
        }
        if (j->has_work()) return j;
    }
    return nullptr;
}
```

`src/device/executor.cpp:373-388` (excerpt)

```cpp
    std::unique_lock lock(mutex_);
    if (job.cooperative) {
        done_cv_.wait(lock, [&] { return active_.empty() && !coop_active_; });
        coop_active_ = true;
        job.joined.assign(threads_.size(), 0);
    } else {
        done_cv_.wait(lock, [&] { return !coop_active_; });
    }
    active_.push_back(&job);
```

A cooperative launch only enters when no other is active, and while it runs (`coop_active_`) no other enters. Inside it, each worker with a sibling enters only once (`joined`), and the launch only ends when all `2 * pairs` workers have entered and left.

## When it applies (and when it does not)

- **Only kernels with `__syncthreads`**, and only for blocks of `L+1` to `2L` threads with `L` of 64, 128, 256 or 512.
- **Only with SMT sibling pairs among the workers.** If the topology has no pair, the launch is not cooperative.
- **It competes with the other variants in the choice by measurement** ([03-automatic-width-selection.md](03-automatic-width-selection.md)). It is not "always better": it wins where register pressure and frequent barriers weigh more than the loss of parallelism within the core.
- **It runs exclusively:** while a cooperative launch is active, no other kernel shares the executor.

### What the microbenchmark measurements showed

Before deciding, the barrier between two threads was measured in isolation:

| where the two threads are | cost of one barrier | overhead with 8 iterations of work between barriers |
|---|---|---|
| SMT siblings of the same core | about 15 ns (about 75 cycles) | 14% |
| different cores of the same CCD | about 36 ns | 35% |
| different CCDs | about 130 ns | 250% |

(CCD, *Core Complex Die*: the chiplet with up to 8 Zen 4 cores and 32 MiB of shared L3; the processor has two. Between cores of the same CCD, the epoch line travels through the L3; between CCDs, it goes through the Infinity Fabric and the I/O die, hence the much higher latency. See [00-basic-concepts.md](../00-basic-concepts.md).) This confirmed that it only makes sense to pair **siblings**: the epoch line never leaves the shared L1D.

The measurement also showed that **two SMT threads doing floating-point computation each take twice as long**: a loop that already saturates the core's vector units gains no throughput from a second thread, which only shares the same pipes. The gain from SMT comes from filling idle cycles, especially those of a thread stalled on a cache miss while the other has ready instructions.

## Results

**Tiled matmul: loses.**

| variant | cycles | wall-clock time |
|---|---|---|
| `128c` | 21.9 Gcyc | 211 ms |
| `256` (whole block) | 14.6 Gcyc | 153 ms |

The reason is the barrier itself. The two siblings stay **in phase**: they arrive together at the tile-loading phase, wait for memory **at the same time** (41% of the cycles are spent in the load phase) and only then compute together. In the `256` variant, each logical thread runs an independent block; the phases of the two siblings become misaligned, and while one is stalled on a cache miss the other occupies the vector units. This is the latency hiding that SMT provides, and that the barrier between siblings eliminates.

**fdtd3d: wins.** This kernel has many live variables and frequent barriers. There `128c` is the best option:

| variant | time |
|---|---|
| `128c` | 2.57 ms |
| `256` | 2.88 ms |
| `32` | 3.20 ms |

With 128 lanes per half, each vector occupies 8 zmm instead of 16, which reduces spill; and the frequent barriers cost ~75 cycles each instead of a full context switch.

**Register spill** (zmm accesses to the stack, tiled matmul): 833 with call, 331 inline, 15 after the static analysis fixes, versus 71 for the 256-lane variant.

## Limitations and trade-offs

- **Synchrony cancels SMT.** The main advantage of SMT, one sibling executing while the other waits for memory, disappears when both are held at the same barrier. Kernels that alternate "load everything" and "compute everything" suffer (tiled matmul).
- **Shared units.** In sections limited by the throughput of the vector units, each sibling runs at half speed; the block does not finish faster for having two threads.
- **Exclusivity.** The cooperative launch does not overlap with other kernels, and workers without a sibling sit idle during it.
- **Busy waiting.** A half that arrives first spins on `pause` until the sibling arrives (without yielding the CPU to the operating system, which is acceptable because the workers are pinned and the expected wait is tens of nanoseconds). If the load of the halves is unequal (one half with many inactive lanes, for example), the wait grows.
- **Contract change.** Two new fields in the context took the ABI to version 8.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): SMT, cores, CCD, registers.
- [02-width-variants.md](../iteration-1/02-width-variants.md): the other variants (32, `s`, whole block).
- [03-automatic-width-selection.md](03-automatic-width-selection.md): how the runtime decides between `128c`, `256`, `32`...
- [09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md): the cost of suspending, which the cooperative variant avoids.
- [10-barrier-pass.md](10-barrier-pass.md): another way of saving context switches.
- [12-static-access-analysis.md](12-static-access-analysis.md) and [13-no-overflow-proof.md](13-no-overflow-proof.md): static analyses of the compiler.
- [17-block-distribution.md](17-block-distribution.md): how blocks are distributed in normal launches.
- [20-concurrent-kernels.md](20-concurrent-kernels.md): the kernel overlap that the cooperative mode turns off.
