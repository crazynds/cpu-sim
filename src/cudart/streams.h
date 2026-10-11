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
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cuda_runtime_api.h"

namespace cpusim::cudart {

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
    bool ready() {
        std::lock_guard lock(mutex);
        return done;
    }
    void finish(cudaError_t e = cudaSuccess) {
        {
            std::lock_guard lock(mutex);
            error = e;
            done = true;
        }
        cv.notify_all();
    }
};
using CompletionPtr = std::shared_ptr<Completion>;

class Queue;

struct EnqueueOptions {
    std::vector<CompletionPtr> after;  // besides the stream's own order
    bool inline_if_idle = false;       // nothing pending before it: run it now, on this thread
    bool sync = false;                 // the caller waits and returns its error itself
    std::function<void()> on_queue;    // called (under the issue lock) only if it is queued
};

class Scheduler {
public:
    Scheduler();
    ~Scheduler();  // runs what is still queued, then stops the threads
    using Options = EnqueueOptions;
    // Queues `work` on `stream` (0, cudaStreamLegacy and cudaStreamPerThread
    // are the default stream) after the ordering rules and `opt.after`.
    CompletionPtr enqueue(cudaStream_t stream, std::function<cudaError_t()> work, Options opt = {});

    // True if nothing `stream` has queued is pending.
    bool idle(cudaStream_t stream);
    void wait_stream(cudaStream_t stream);
    void wait_all();

    Queue* create(unsigned flags);  // a new user stream
    void destroy(Queue* q);         // after its work is done

    // The first error of asynchronous work since the last call, cleared (or not).
    cudaError_t take_error();
    cudaError_t peek_error();

private:
    friend class Queue;
    Queue* resolve(cudaStream_t stream);
    void record_error(cudaError_t e);

    std::mutex mutex_;  // issue order across streams
    std::unique_ptr<Queue> legacy_;
    std::vector<Queue*> streams_;  // user streams (owned through their handles)
    cudaError_t error_ = cudaSuccess;
    std::mutex error_mutex_;
};

Scheduler& scheduler();

// Host memory from cudaMallocHost / cudaHostAlloc: copies from it are not
// staged (like CUDA, the program must not change it before the copy runs).
void pinned_add(const void* p, size_t size);
void pinned_remove(const void* p);
bool pinned_contains(const void* p, size_t size);

}  // namespace cpusim::cudart

// The stream handle is the queue.
struct CUstream_st {
    cpusim::cudart::Queue* queue = nullptr;
};
