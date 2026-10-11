# 20. Concurrent kernels (asynchronous streams)

> **In one sentence:** a kernel launch (`<<<...>>>`) now returns to the program immediately, each CUDA stream became a queue with its own thread, and the executor accepts several active kernels at the same time; as a result, kernels from different streams run together, the end of one kernel overlaps the start of the next, and two small kernels in different streams take the time of one.

The project's general concepts (host, device, worker, the `cpusimd` daemon) are in [00-basic-concepts.md](../00-basic-concepts.md). The CUDA stream semantics that the runtime has to reproduce:

- **Stream**: a sequence of device operations (launches, copies, event records, host callbacks) with a total order within the stream and no guaranteed order between streams. Launches and `cudaMemcpyAsync` return to the host before the operation executes; the host only synchronizes when it asks to (`cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaEventSynchronize`) or when it calls a synchronous API, such as `cudaMemcpy` without `Async`.
- **Legacy default stream (stream 0)**: with the legacy semantics (the `nvcc` default, without `--default-stream per-thread`), stream 0 is implicitly synchronizing: an operation on it only starts after all work already issued to all **blocking** streams of the context has finished, and an operation issued after it to any blocking stream only starts after it finishes. Streams created with `cudaStreamNonBlocking` are outside both rules.
- **Event**: `cudaEventRecord(e, s)` inserts into `s` a marker that completes when all earlier work in `s` finishes. `cudaStreamWaitEvent(s2, e)` makes the work issued to `s2` from that point on wait for the **most recent record** of `e` at the time of the call; later records do not affect that wait.
- **Asynchronous errors**: an error in the execution of a kernel is not returned by the launch, but by a later call (the next synchronization or `cudaGetLastError`).

## The problem

Previously, two things made the simulator waste CPUs.

**1. The launch blocked the program.** On a GPU, `kernel<<<grid, block>>>(...)` only places the kernel in a queue and returns immediately; the program can prepare the next work while the GPU computes. In the simulator, the call only returned when the kernel finished. Streams existed in name only: everything ran in sequence.

**2. The executor ran one launch at a time.** Even if two kernels were ready, the second waited for the first to finish completely. And the end of any kernel has a "tail": the last blocks run on few workers while the others have nothing left to do.

```text
one launch at a time (28 workers):
time ->
kernel A: ##############################........   <- tail: few workers busy
kernel B:                                       ##############################....
```

An extreme case is the test `tests/cuda/async.cu`: a kernel with only **2 blocks** that lasts about 60 ms. It occupies 2 of the 28 workers. Two of these kernels, in different streams, should run together (4 workers busy, 60 ms in total), but they took 120 ms.

## The idea

Copy the GPU model, in two layers:

**In the runtime (host side):** each stream is a FIFO queue consumed by a host thread of its own. The application thread only enqueues the operation and returns. The stream thread pops the operations **in order**, waits for the dependencies of each one and submits it to the executor synchronously (the call only returns when the kernel finishes). Threads of different streams submit in parallel, which is the equivalent of the multiple hardware queues (work queues) of a GPU.

**In the executor:** instead of one job at a time, there is a list of active jobs. A free worker takes blocks from the **oldest job that still has undelivered blocks**. When job A only has blocks in execution (the tail), the workers that became free already start consuming job B, as a GPU's block scheduler does when dispatching blocks of a following kernel to SMs that became free.

```text
several active launches (28 workers):
kernel A: ##############################....
kernel B:                           ######################################
                                    ^ starts while A finishes
```

The CUDA ordering rules become **dependencies** between tasks: "this task only starts when that one finishes".

## How it works, step by step

### The ordering rules

| CUDA rule | How it becomes a dependency |
|---|---|
| Tasks of the same stream run in order | each task depends on the last one of its stream |
| The default stream waits for all "blocking" streams | a task on stream 0 depends on the last one of each blocking stream |
| Blocking streams wait for the default stream | a task on a blocking stream depends on the last one of stream 0 |
| `cudaStreamWaitEvent(s, e)` | an empty task on `s` that depends on the record of `e` |

A stream is "blocking" unless it was created with the `cudaStreamNonBlocking` option.

### A reduced example

The program does:

```c
k1<<<..., s1>>>();      // A
k2<<<..., s2>>>();      // B
k3<<<..., s1>>>();      // C
cudaDeviceSynchronize();
```

1. `A` enters the queue of `s1`. It depends on nothing (dependencies that already finished are discarded). The thread of `s1` wakes up and calls the executor. The program has already moved on.
2. `B` enters the queue of `s2`. It does not depend on `A` (different streams, neither is the default). The thread of `s2` calls the executor; now there are two active jobs.
3. `C` enters the queue of `s1`, after `A`. The thread of `s1` is busy waiting for `A`; when `A` finishes, it takes `C`.
4. `cudaDeviceSynchronize` waits for the last task of each stream.

### Memory copies

A host-to-device copy (`cudaMemcpyAsync`) from **pageable** memory has an implicit guarantee in CUDA: the driver copies the source to a pinned staging buffer before returning (DMA cannot read pageable memory), so the program can rewrite the source buffer as soon as the call returns. If the copy were merely placed in the queue pointing to the source, and the program rewrote the buffer before it executed, we would copy wrong data.

The adopted rule:

- **Idle queue** (no pending dependency): the copy runs **immediately**, on the application thread, which preserves the stream order without staging. This is the most common case and has no extra cost.
- **Busy queue**: the data is copied to an intermediate buffer (**staging**) at the time of the call, and the actual copy uses that buffer later.
- **Exception:** if the source is **pinned** (`cudaMallocHost`/`cudaHostAlloc`) or **managed** memory, there is no staging. As in CUDA, where DMA reads directly from these regions, it is up to the program not to modify them before the copy completes.

### Measuring utilization

With several kernels at the same time, "busy device time" cannot be the sum of the kernel durations (it would count overlapping intervals twice). The metric became the measure of the **union of the execution intervals**: an atomic counter of running kernels; the transition from 0 to 1 records the instant, the one from 1 to 0 adds up the interval.

### In the daemon

When the program uses the `cpusimd` daemon, each call is a synchronous request on a socket. If all streams shared one connection, a launch on `s2` would sit behind the response to a launch on `s1` (head-of-line blocking), serializing the streams again. For this reason, the process opens **extra connections** when needed, and they **attach to the same context** (the same memory space, the same modules) with the `kHelloAttach` option. The daemon only accepts the attachment if the request comes from the same process that owns the context.

## The code

### The header that summarizes the rules

`src/cudart/streams.h:1-10`

```cpp
#pragma once
// Asynchronous streams for libcpusim_cudart. Every stream is a FIFO of
// operations run by a host thread of its own, in issue order; operations of
// different streams run concurrently (the device takes several launches at
// once). The CUDA ordering rules become dependencies between operations:
//   - the legacy default stream (0) waits for all work already issued to the
//     blocking streams, and blocking streams wait for work already issued to it;
//   - cudaStreamWaitEvent makes a stream wait for an event's last record.
// Errors of asynchronous work are kept and returned by the next
// synchronization (or cudaGetLastError), as in CUDA.
```

The last sentence corresponds to the CUDA asynchronous error semantics: the execution error of a kernel does not appear at the launch, but at the next synchronization.

### `Completion` and `EnqueueOptions`

`src/cudart/streams.h:23-55`

```cpp
// The end of one operation; others (and the host) wait on it.
struct Completion {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    cudaError_t error = cudaSuccess;  // of this operation
    void wait() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return done; });
    }
    // ...
};
using CompletionPtr = std::shared_ptr<Completion>;

class Queue;

struct EnqueueOptions {
    std::vector<CompletionPtr> after;  // besides the stream's own order
    bool inline_if_idle = false;       // nothing pending before it: run it now, on this thread
    bool sync = false;                 // the caller waits and returns its error itself
    std::function<void()> on_queue;    // called (under the issue lock) only if it is queued
};
```

- `Completion` is the completion signal of an operation (a shared future, with the error code): whoever depends on it calls `wait()`.
- `after`: extra dependencies (used by `cudaStreamWaitEvent`).
- `inline_if_idle`: "if nothing is pending, run now on this thread" (used by copies).
- `on_queue`: an action that only happens if the task goes to the queue (used for staging).

### Enqueueing with the ordering rules

`src/cudart/streams.cpp:92-129`

```cpp
CompletionPtr Scheduler::enqueue(cudaStream_t stream, std::function<cudaError_t()> work, Options opt) {
    Op op;
    op.work = std::move(work);
    op.deps = std::move(opt.after);
    op.done = std::make_shared<Completion>();
    op.sync = opt.sync;
    std::unique_lock lock(mutex_);
    Queue* q = resolve(stream);
    auto add = [&](const CompletionPtr& c) {
        if (c && !c->ready()) op.deps.push_back(c);
    };
    add(q->last());
    if (q == legacy_.get()) {
        // The legacy default stream waits for every blocking stream...
        for (Queue* s : streams_)
            if (s->blocking()) add(s->last());
    } else if (q->blocking() && legacy_) {
        // ...and blocking streams wait for it.
        add(legacy_->last());
    }
    // ...
    if (opt.inline_if_idle && op.deps.empty()) {
        // Nothing pending: run it now on this thread, still in issue order
        // (later operations depend on it through last()).
        CompletionPtr done = op.done;
        q->set_last(done);
        lock.unlock();
        cudaError_t e = op.work ? op.work() : cudaSuccess;
        if (e != cudaSuccess && !op.sync) record_error(e);
        done->finish(e);
        return done;
    }
    if (opt.on_queue) opt.on_queue();
    CompletionPtr done = op.done;
    q->push(std::move(op));
    return done;
}
```

The three calls to `add` are the first three rows of the rules table. Even when the task runs immediately (`inline_if_idle`), it becomes the "last" of the stream (`set_last`), so that the following ones stay in order.

### The thread of each stream

`src/cudart/streams.cpp:60-75`

```cpp
void Queue::run() {
    while (true) {
        Op op;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || !ops_.empty(); });
            if (ops_.empty()) return;  // stopping, and everything ran
            op = std::move(ops_.front());
            ops_.pop_front();
        }
        for (auto& d : op.deps) d->wait();
        cudaError_t e = op.work ? op.work() : cudaSuccess;
        if (e != cudaSuccess && !op.sync) sched_.record_error(e);
        op.done->finish(e);
    }
}
```

The stream thread's loop: pop the next operation, block on the dependencies, execute and signal the `Completion`. The thread is only created on the first `push` (`streams.cpp:30`), so a stream that is never used does not cost a thread.

### The asynchronous launch

`src/cudart/runtime.cpp:1000-1012`

```cpp
    Profiler& prof = profiler();
    if (!prof.enabled && !debug) {
        // Queued on its stream; returns right away.
        auto buf = std::make_shared<std::vector<uint8_t>>(std::move(params));
        runtime().sched.enqueue(stream, [d, function, cfg, buf, name = k.name]() -> cudaError_t {
            cpusim::LaunchResult result;
            Error e = d->launch(function, cfg, buf->data(), buf->size(), &result);
            if (e != Error::Success && result.report[0])
                std::fprintf(stderr, "cpusim: kernel %s failed:\n  %s\n", name.c_str(), result.report);
            return static_cast<cudaError_t>(e);
        });
        return cudaSuccess;
    }
```

The kernel parameters are copied into `buf` (the program may change the original variables right afterwards). With profiling or debug mode on, the launch remains synchronous (lines 989-998), because these modes need the result right away.

### Copies and staging

`src/cudart/runtime.cpp:656-670`

```cpp
cpusim::cudart::CompletionPtr queue_copy(cpusim::Device* d, void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                         cudaStream_t stream, bool sync) {
    kind = resolve_kind(d, dst, src, kind);
    auto staged = std::make_shared<std::vector<uint8_t>>();
    const bool host_src = kind == cudaMemcpyHostToDevice || kind == cudaMemcpyHostToHost;
    const bool stage = host_src && !sync && !d->is_managed_pointer(src) && !cpusim::cudart::pinned_contains(src, count);
    cpusim::cudart::EnqueueOptions opt;
    opt.inline_if_idle = true;
    opt.sync = sync;
    if (stage) opt.on_queue = [staged, src, count] { staged->assign(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + count); };
    return runtime().sched.enqueue(
        stream,
        [d, dst, src, count, kind, staged] { return copy_now(d, dst, staged->empty() ? src : staged->data(), count, kind); },
        std::move(opt));
}
```

`stage` is true for a host source, an asynchronous copy, and memory that is neither managed nor pinned. Even so, staging only happens if the task goes to the queue (`on_queue`); if the queue was idle, the copy runs immediately directly from the source.

### `cudaStreamWaitEvent`

`src/cudart/runtime.cpp:768-777`

```cpp
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int) {
    if (!event) return record(Error::InvalidResourceHandle);
    cpusim::cudart::EnqueueOptions opt;
    {
        std::lock_guard lock(event->mutex);
        if (event->pending) opt.after.push_back(event->pending);
    }
    runtime().sched.enqueue(stream, nullptr, std::move(opt));
    return cudaSuccess;
}
```

An empty task (`nullptr`) that depends on the last record of the event. Since everything that comes later in the stream depends on it, the entire stream starts waiting for the event.

### The executor with several jobs

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

`active_` is in arrival order. A free worker walks the list and takes the first job that still has blocks to deliver. Thus, older kernels have priority, but nobody sits idle if a newer one has work.

`src/device/executor.cpp:373-383`

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
    start_cv_.notify_all();
    done_cv_.wait(lock, [&] { return job.done; });
```

Each stream thread that calls `run` puts its job in the list and waits for it to finish. A cooperative job ([15-smt-cooperative-variant.md](15-smt-cooperative-variant.md)) is the exception: it needs all pairs of sibling CPUs, so it runs alone (it waits for the list to empty, and nobody enters while it runs).

### The "some kernel running" accounting

`src/device/device_core.cpp:424-437`

```cpp
        // Busy time = time with at least one kernel running (launches from
        // several streams overlap). `since` lets the monitor count a long
        // kernel while it runs, not only when it ends.
        struct BusyPeriod {
            DeviceCounters& c;
            explicit BusyPeriod(DeviceCounters& counters) : c(counters) {
                if (c.kernels_running.fetch_add(1) == 0) c.kernel_since_ns.store(monotonic_ns());
            }
            ~BusyPeriod() {
                if (c.kernels_running.fetch_sub(1) != 1) return;
                uint64_t since = c.kernel_since_ns.exchange(0);
                if (since) c.kernel_busy_ns.fetch_add(monotonic_ns() - since);
            }
        } busy{counters_};
```

Whoever moves the counter from 0 to 1 records the time; whoever moves it from 1 back to 0 adds up the period.

### Extra connections in the daemon

`src/device/protocol.h:52-55`

```cpp
// More connections of the same process join its context (Hello's c = the
// context id): one per stream in flight, so streams do not wait for each
// other on the socket. Closing one of them leaves the context open.
constexpr uint64_t kHelloAttach = 2;
```

`src/driver/server.cpp:240-250`

```cpp
                if (req.b & proto::kHelloAttach) {
                    // Only the process that owns the context may join it.
                    bool owner = false;
                    for (const auto& info : core_.contexts()) owner |= info.id == req.c && info.pid == cred.pid;
                    if (!owner) {
                        h.header.error = static_cast<int32_t>(Error::InvalidValue);
                        send(fd, &h, sizeof(h), MSG_NOSIGNAL);
                        goto out;
                    }
                    ctx = uint32_t(req.c);
                    attached = true;
```

The daemon checks, through the socket credentials (`cred.pid`), that the requester of the attachment is the process that owns the context. On the client side (`src/device/daemon_client.cpp:290-316`), an idle connection is reused; if there is none, a new one is opened with `kHelloAttach`; if even that is not possible, it waits for the main connection.

## When it applies (and when it does not)

- **Ordinary launches:** always asynchronous.
- **With profiling or debugging (`CPUSIM_DEBUG`):** synchronous, waiting for everything that came before.
- **Real overlap** only happens between kernels that **can** run together under the rules: different streams, with no dependency between them. Programs that only use the default stream gain the asynchronous launch (the program keeps working while the kernel runs), but the kernels remain in sequence.
- **Within a single stream** there is no tail overlap: the second kernel is only handed to the executor when the first finishes, as the stream order requires. Tail overlap happens only between different streams.
- **Cooperative kernels** run alone.

## Results

Test `tests/cuda/async.cu`: two kernels of 2 blocks, each calibrated to about 60 ms.

| Where | Two streams | Same stream (twice) | One kernel alone |
|---|---|---|---|
| RTX 5060 GPU (reference) | 59.9 ms | 119.6 ms | |
| Simulator, in-process | 60.7 ms | 119.1 ms | 59.5 ms |
| Simulator, via daemon | 64.0 ms | 119.4 ms | |

The launch now returns in about **0 ms**. The behavior became the same as the GPU's: two streams take the time of one kernel; the same stream takes twice as long. Via the daemon there are a few extra milliseconds, from the socket path.

The test also checks that a host callback (`cudaLaunchHostFunc`) only runs after the previous kernel in its stream, and that events measure the time of a kernel.

## Limitations and trade-offs

- **One thread per used stream.** Programs that create hundreds of streams create hundreds of threads (asleep almost all the time).
- **Staging costs an extra copy** when the queue is busy and the source is ordinary memory. It is the same as what a CUDA driver does with pageable memory.
- **Split cache.** Two kernels at the same time share the workers and the caches. The "same block, same core" affinity of [17-block-distribution.md](17-block-distribution.md) becomes weaker in this case.
- **Width measurement.** The slice-based measurement uses the workers' time per block, not the clock, precisely so as not to be fooled by other launches running at the same time ([03-automatic-width-selection.md](03-automatic-width-selection.md)).

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): host, device, kernel, worker.
- [03-automatic-width-selection.md](03-automatic-width-selection.md): why the measurement uses worker time.
- [15-smt-cooperative-variant.md](15-smt-cooperative-variant.md): the type of job that runs alone.
- [16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md): another way to occupy idle workers.
- [17-block-distribution.md](17-block-distribution.md): how the blocks of each job are delivered.
