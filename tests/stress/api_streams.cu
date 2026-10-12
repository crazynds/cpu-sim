// Streams and events: a multi-stream copy/compute pipeline, cross-stream
// dependencies with cudaStreamWaitEvent, event timing, cudaStreamQuery while
// a kernel is held, host callbacks (cudaLaunchHostFunc, cudaStreamAddCallback),
// legacy default stream ordering, non-blocking streams, priorities and
// cudaStreamPerThread.
#include "stress.cuh"

#include <atomic>

__global__ void slow_fill(int* p, int n, int value, int spin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Busy work so a missing dependency shows up as a wrong result.
    unsigned x = unsigned(i);
    for (int k = 0; k < spin; ++k) x = x * 1664525u + 1013904223u;
    p[i] = x == 0x9e3779b9u ? -1 : value;
}

__global__ void read_plus(const int* src, int* dst, int n, int add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] + add;
}

__global__ void chunk_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        out[i] = v * v + 2.0f * v + 1.0f;
    }
}

// Waits until the host sets *flag (mapped host memory), with a bound so a
// runtime that runs the kernel inside the launch call cannot hang forever.
__global__ void wait_for_flag(volatile int* flag, int* seen) {
    long long tries = 0;
    while (*flag == 0 && tries < (1ll << 24)) ++tries;
    *seen = *flag;
}

struct CallbackLog {
    std::atomic<int> next{0};
    int order[16];
    int observed[16];
    const int* pinned;
};
struct CallbackArg {
    CallbackLog* log;
    int id;
};

static void CUDART_CB host_fn(void* p) {
    auto* a = static_cast<CallbackArg*>(p);
    int slot = a->log->next.fetch_add(1);
    a->log->order[slot] = a->id;
    a->log->observed[slot] = a->log->pinned[a->id];  // the D2H copy before it in the stream has finished
}

static void CUDART_CB legacy_cb(cudaStream_t, cudaError_t status, void* p) {
    auto* a = static_cast<CallbackArg*>(p);
    int slot = a->log->next.fetch_add(1);
    a->log->order[slot] = status == cudaSuccess ? a->id : -1000;
    a->log->observed[slot] = a->log->pinned[a->id];
}

int main() {
    using namespace stress;

    {  // 4-stream pipeline: H2D, kernel, D2H per chunk
        const int n = 1 << 22, streams = 4, chunks = 16, cn = n / chunks;
        float *h_in, *h_out, *d_in, *d_out;
        STRESS_CHECK(cudaMallocHost(&h_in, n * sizeof(float)));
        STRESS_CHECK(cudaMallocHost(&h_out, n * sizeof(float)));
        STRESS_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        STRESS_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        auto x = random_floats(n, 1);
        std::memcpy(h_in, x.data(), n * sizeof(float));
        cudaStream_t s[streams];
        for (auto& st : s) STRESS_CHECK(cudaStreamCreate(&st));
        double k = best_ms([&] {
            for (int c = 0; c < chunks; ++c) {
                cudaStream_t st = s[c % streams];
                size_t o = size_t(c) * cn;
                STRESS_CHECK(cudaMemcpyAsync(d_in + o, h_in + o, cn * sizeof(float), cudaMemcpyHostToDevice, st));
                chunk_kernel<<<(cn + 255) / 256, 256, 0, st>>>(d_in + o, d_out + o, cn);
                STRESS_CHECK(cudaMemcpyAsync(h_out + o, d_out + o, cn * sizeof(float), cudaMemcpyDeviceToHost, st));
            }
            for (auto& st : s) STRESS_CHECK(cudaStreamSynchronize(st));
        });
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = x[i] * x[i] + 2.0f * x[i] + 1.0f;
        });
        bool ok = equal(h_out, want.data(), n);
        for (auto& st : s) STRESS_CHECK(cudaStreamDestroy(st));
        STRESS_CHECK(cudaFreeHost(h_in));
        STRESS_CHECK(cudaFreeHost(h_out));
        STRESS_CHECK(cudaFree(d_in));
        STRESS_CHECK(cudaFree(d_out));
        report("multi_stream_pipeline", ok, k, ref);
    }

    {  // stream B waits for an event recorded in stream A
        const int n = 1 << 16;
        Buffer<int> a(n), b(n);
        cudaStream_t sa, sb;
        STRESS_CHECK(cudaStreamCreate(&sa));
        STRESS_CHECK(cudaStreamCreate(&sb));
        cudaEvent_t ev;
        STRESS_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        bool ok = true;
        double t0 = now_ms();
        for (int round = 1; round <= 5 && ok; ++round) {
            slow_fill<<<n / 256, 256, 0, sa>>>(a, n, round * 100, 2000);
            STRESS_CHECK(cudaEventRecord(ev, sa));
            STRESS_CHECK(cudaStreamWaitEvent(sb, ev, 0));
            read_plus<<<n / 256, 256, 0, sb>>>(a, b, n, round);
            STRESS_CHECK(cudaStreamSynchronize(sb));
            auto got = b.download();
            for (int i = 0; i < n && ok; ++i) ok = got[i] == round * 101;
        }
        double k = now_ms() - t0;
        STRESS_CHECK(cudaEventDestroy(ev));
        STRESS_CHECK(cudaStreamDestroy(sa));
        STRESS_CHECK(cudaStreamDestroy(sb));
        report("stream_wait_event", ok, k, -1);
    }

    {  // event timing: elapsed time positive, ordered, cudaEventQuery / cudaEventSynchronize
        const int n = 1 << 16;
        Buffer<int> a(n);
        cudaEvent_t e0, e1, e2;
        STRESS_CHECK(cudaEventCreate(&e0));
        STRESS_CHECK(cudaEventCreate(&e1));
        STRESS_CHECK(cudaEventCreateWithFlags(&e2, cudaEventBlockingSync));
        STRESS_CHECK(cudaEventRecord(e0));
        slow_fill<<<n / 256, 256>>>(a, n, 1, 5000);
        STRESS_CHECK(cudaEventRecord(e1));
        slow_fill<<<n / 256, 256>>>(a, n, 2, 5000);
        STRESS_CHECK(cudaEventRecord(e2));
        STRESS_CHECK(cudaEventSynchronize(e2));
        float t01 = -1, t02 = -1, t12 = -1;
        STRESS_CHECK(cudaEventElapsedTime(&t01, e0, e1));
        STRESS_CHECK(cudaEventElapsedTime(&t02, e0, e2));
        STRESS_CHECK(cudaEventElapsedTime(&t12, e1, e2));
        cudaError_t q = cudaEventQuery(e1);
        bool ok = q == cudaSuccess && t01 > 0 && t12 > 0 && t02 >= t01 && std::fabs(t02 - (t01 + t12)) < 0.01f + 0.01f * t02 &&
                  t02 < 2000.0f;
        if (!ok) std::printf("  t01=%g t02=%g t12=%g q=%d\n", t01, t02, t12, int(q));
        auto got = a.download();
        for (int i = 0; i < n && ok; ++i) ok = got[i] == 2;
        STRESS_CHECK(cudaEventDestroy(e0));
        STRESS_CHECK(cudaEventDestroy(e1));
        STRESS_CHECK(cudaEventDestroy(e2));
        report("event_elapsed_time", ok, t02, -1);
    }

    {  // cudaStreamQuery: not ready while a kernel waits for the host, ready after
        int* flag;
        int* seen;
        STRESS_CHECK(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
        STRESS_CHECK(cudaHostAlloc(&seen, sizeof(int), cudaHostAllocMapped));
        *flag = 0;
        *seen = -1;
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        bool ok = cudaStreamQuery(s) == cudaSuccess;  // empty stream
        double t0 = now_ms();
        wait_for_flag<<<1, 1, 0, s>>>(flag, seen);
        STRESS_CHECK(cudaGetLastError());
        cudaError_t busy = cudaStreamQuery(s);
        cudaGetLastError();  // cudaErrorNotReady is not an error state; clear it anyway
        reinterpret_cast<volatile int*>(flag)[0] = 1;
        STRESS_CHECK(cudaStreamSynchronize(s));
        double k = now_ms() - t0;
        cudaError_t done = cudaStreamQuery(s);
        ok = ok && busy == cudaErrorNotReady && done == cudaSuccess && *seen == 1;
        if (!ok) std::printf("  busy=%d done=%d seen=%d\n", int(busy), int(done), *seen);
        STRESS_CHECK(cudaStreamDestroy(s));
        STRESS_CHECK(cudaFreeHost(flag));
        STRESS_CHECK(cudaFreeHost(seen));
        report("stream_query", ok, k, -1);
    }

    {  // host callbacks run in stream order, after the work before them
        const int n = 8;
        int* pinned;
        STRESS_CHECK(cudaMallocHost(&pinned, n * sizeof(int)));
        for (int i = 0; i < n; ++i) pinned[i] = -1;
        Buffer<int> d(n);
        CallbackLog log;
        log.pinned = pinned;
        CallbackArg args[n];
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        double t0 = now_ms();
        for (int i = 0; i < n; ++i) {
            args[i] = {&log, i};
            slow_fill<<<1, n, 0, s>>>(d, n, 10 * (i + 1), 1000);
            STRESS_CHECK(cudaMemcpyAsync(pinned + i, d.p + i, sizeof(int), cudaMemcpyDeviceToHost, s));
            if (i % 2 == 0)
                STRESS_CHECK(cudaLaunchHostFunc(s, host_fn, &args[i]));
            else
                STRESS_CHECK(cudaStreamAddCallback(s, legacy_cb, &args[i], 0));
        }
        STRESS_CHECK(cudaStreamSynchronize(s));
        double k = now_ms() - t0;
        bool ok = log.next.load() == n;
        for (int i = 0; i < n && ok; ++i) ok = log.order[i] == i && log.observed[i] == 10 * (i + 1);
        if (!ok)
            for (int i = 0; i < n; ++i) std::printf("  slot %d: order %d observed %d\n", i, log.order[i], log.observed[i]);
        STRESS_CHECK(cudaStreamDestroy(s));
        STRESS_CHECK(cudaFreeHost(pinned));
        report("host_callbacks_order", ok, k, -1);
    }

    {  // legacy default stream: work in blocking streams is ordered with stream 0
        const int n = 1 << 16;
        Buffer<int> a(n), b(n), c(n);
        cudaStream_t s1, s2;
        STRESS_CHECK(cudaStreamCreate(&s1));
        STRESS_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamDefault));
        bool ok = true;
        double t0 = now_ms();
        for (int round = 1; round <= 4 && ok; ++round) {
            slow_fill<<<n / 256, 256, 0, s1>>>(a, n, round, 3000);  // s1
            read_plus<<<n / 256, 256>>>(a, b, n, 10);               // stream 0: waits for s1
            read_plus<<<n / 256, 256, 0, s2>>>(b, c, n, 100);       // s2: waits for stream 0
            STRESS_CHECK(cudaStreamSynchronize(s2));
            auto got = c.download();
            for (int i = 0; i < n && ok; ++i) ok = got[i] == round + 110;
        }
        double k = now_ms() - t0;
        STRESS_CHECK(cudaStreamDestroy(s1));
        STRESS_CHECK(cudaStreamDestroy(s2));
        report("legacy_default_stream_order", ok, k, -1);
    }

    {  // non-blocking stream: flags reported, explicit event ordering works
        const int n = 1 << 16;
        Buffer<int> a(n), b(n);
        cudaStream_t nb;
        STRESS_CHECK(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking));
        unsigned flags = 99, dflags = 99;
        STRESS_CHECK(cudaStreamGetFlags(nb, &flags));
        STRESS_CHECK(cudaStreamGetFlags(0, &dflags));
        cudaEvent_t ev;
        STRESS_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        double t0 = now_ms();
        slow_fill<<<n / 256, 256>>>(a, n, 77, 2000);  // stream 0
        STRESS_CHECK(cudaEventRecord(ev, 0));
        STRESS_CHECK(cudaStreamWaitEvent(nb, ev, 0));
        read_plus<<<n / 256, 256, 0, nb>>>(a, b, n, 1);
        STRESS_CHECK(cudaStreamSynchronize(nb));
        double k = now_ms() - t0;
        auto got = b.download();
        bool ok = flags == cudaStreamNonBlocking && dflags == cudaStreamDefault;
        if (!ok) std::printf("  flags=%u default=%u\n", flags, dflags);
        for (int i = 0; i < n && ok; ++i) ok = got[i] == 78;
        STRESS_CHECK(cudaEventDestroy(ev));
        STRESS_CHECK(cudaStreamDestroy(nb));
        report("non_blocking_stream", ok, k, -1);
    }

    {  // stream priorities and cudaStreamPerThread
        int lo = 0, hi = 0;
        STRESS_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        cudaStream_t sl, sh;
        STRESS_CHECK(cudaStreamCreateWithPriority(&sl, cudaStreamNonBlocking, lo));
        STRESS_CHECK(cudaStreamCreateWithPriority(&sh, cudaStreamNonBlocking, hi));
        int pl = 1234, ph = 1234;
        STRESS_CHECK(cudaStreamGetPriority(sl, &pl));
        STRESS_CHECK(cudaStreamGetPriority(sh, &ph));
        const int n = 1 << 14;
        Buffer<int> a(n), b(n), c(n);
        double t0 = now_ms();
        slow_fill<<<n / 256, 256, 0, sl>>>(a, n, 5, 100);
        slow_fill<<<n / 256, 256, 0, sh>>>(b, n, 6, 100);
        STRESS_CHECK(cudaStreamSynchronize(sl));
        STRESS_CHECK(cudaStreamSynchronize(sh));
        read_plus<<<n / 256, 256, 0, cudaStreamPerThread>>>(a, c, n, 1000);
        STRESS_CHECK(cudaMemcpyAsync(b.p, c.p, n * sizeof(int), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        STRESS_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        double k = now_ms() - t0;
        bool ok = hi <= lo && pl == lo && ph == hi;
        if (!ok) std::printf("  range [%d,%d] got %d %d\n", lo, hi, pl, ph);
        auto gb = b.download();
        for (int i = 0; i < n && ok; ++i) ok = gb[i] == 1005;
        STRESS_CHECK(cudaStreamDestroy(sl));
        STRESS_CHECK(cudaStreamDestroy(sh));
        report("priorities_per_thread_stream", ok, k, -1);
    }

    {  // cudaMemsetAsync / cudaMemcpyAsync on pageable memory in a stream, ordered with kernels
        const int n = 1 << 18;
        std::vector<int> host(n, 0);
        Buffer<int> a(n), b(n);
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        double t0 = now_ms();
        STRESS_CHECK(cudaMemsetAsync(a.p, 0, n * sizeof(int), s));
        read_plus<<<n / 256, 256, 0, s>>>(a, b, n, 42);
        STRESS_CHECK(cudaMemsetAsync(a.p, 0xFF, 64 * sizeof(int), s));
        read_plus<<<n / 256, 256, 0, s>>>(a, a, n, 1);  // in place: -1 + 1 = 0 for the first 64
        STRESS_CHECK(cudaMemcpyAsync(host.data(), b.p, n * sizeof(int), cudaMemcpyDeviceToHost, s));
        STRESS_CHECK(cudaStreamSynchronize(s));
        double k = now_ms() - t0;
        auto ga = a.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) ok = host[i] == 42 && ga[i] == (i < 64 ? 0 : 1);
        STRESS_CHECK(cudaStreamDestroy(s));
        report("async_memset_memcpy_order", ok, k, -1);
    }

    return finish();
}
