// Asynchronous streams: launches return before the kernel ends, kernels of
// different streams run at the same time (two small kernels take about as
// long as one) while one stream stays in order, host callbacks run after the
// work before them, and events time a kernel. Timing thresholds are loose.
#include <atomic>
#include <chrono>
#include <vector>

#include "check.cuh"

// Busy work whose length scales with `iters`; the result depends on it so it
// cannot be skipped.
__global__ void spin(float* out, long long iters) {
    float v = threadIdx.x * 0.001f;
    for (long long i = 0; i < iters; ++i) v = v * 0.999999f + 0.000001f;
    out[blockIdx.x * blockDim.x + threadIdx.x] = v;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::atomic<int> callback_value{0};
static void after_kernel(void*) { callback_value = 1; }  // must not touch device memory

int main() {
    Checker c("async");
    float* out;
    CHECK_CUDA(cudaMalloc(&out, 4096 * sizeof(float)));
    cudaStream_t s1, s2;
    CHECK_CUDA(cudaStreamCreate(&s1));
    CHECK_CUDA(cudaStreamCreate(&s2));

    // Calibrate: a 2-block kernel of about 60 ms (whatever the machine).
    long long iters = 1 << 16;
    double t = 0;
    for (int tries = 0; tries < 20; ++tries) {
        spin<<<2, 32, 0, s1>>>(out, iters);  // also compiles it (JIT) on the first round
        CHECK_CUDA(cudaStreamSynchronize(s1));
        double t0 = now_ms();
        spin<<<2, 32, 0, s1>>>(out, iters);
        CHECK_CUDA(cudaStreamSynchronize(s1));
        t = now_ms() - t0;
        if (t > 40) break;
        iters = t > 1 ? (long long)(iters * 60.0 / t) : iters * 16;
    }
    std::printf("  one kernel: %.1f ms\n", t);
    c.expect(t > 20, "calibration reached a measurable kernel", -1, t, 20);

    // 1. The launch call returns before the kernel is done.
    double t0 = now_ms();
    spin<<<2, 32, 0, s1>>>(out, iters);
    double issue = now_ms() - t0;
    CHECK_CUDA(cudaStreamSynchronize(s1));
    std::printf("  launch returned after %.2f ms\n", issue);
    c.expect(issue < 0.5 * t, "launch is asynchronous", -1, issue, 0.5 * t);

    // 2. Two streams at once, then the same stream twice.
    t0 = now_ms();
    spin<<<2, 32, 0, s1>>>(out, iters);
    spin<<<2, 32, 0, s2>>>(out + 2048, iters);
    CHECK_CUDA(cudaStreamSynchronize(s1));
    CHECK_CUDA(cudaStreamSynchronize(s2));
    double both = now_ms() - t0;
    t0 = now_ms();
    spin<<<2, 32, 0, s1>>>(out, iters);
    spin<<<2, 32, 0, s1>>>(out + 2048, iters);
    CHECK_CUDA(cudaStreamSynchronize(s1));
    double serial = now_ms() - t0;
    std::printf("  two streams: %.1f ms, one stream twice: %.1f ms\n", both, serial);
    c.expect(both < 1.6 * t, "kernels of two streams overlap", -1, both, 1.6 * t);
    c.expect(serial > 1.7 * t, "one stream runs in order", -1, serial, 1.7 * t);

    // 3. A host callback runs after the kernel before it in its stream.
    callback_value = 0;
    spin<<<2, 32, 0, s2>>>(out, iters / 4);
    CHECK_CUDA(cudaLaunchHostFunc(s2, after_kernel, nullptr));
    c.expect(callback_value == 0, "callback waits for the kernel", -1, callback_value, 0);
    CHECK_CUDA(cudaStreamSynchronize(s2));
    c.expect(callback_value == 1, "callback ran", -1, callback_value, 1);

    // 4. Events around a kernel.
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventRecord(e0, s1));
    spin<<<2, 32, 0, s1>>>(out, iters);
    CHECK_CUDA(cudaEventRecord(e1, s1));
    c.expect(cudaEventQuery(e1) == cudaErrorNotReady, "event not reached yet");
    CHECK_CUDA(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    std::printf("  events: %.1f ms\n", ms);
    c.expect(ms > 0.5 * t, "events time the kernel", -1, ms, 0.5 * t);

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaStreamDestroy(s1));
    CHECK_CUDA(cudaStreamDestroy(s2));
    CHECK_CUDA(cudaFree(out));
    CHECK_CUDA(cudaGetLastError());
    return c.finish();
}
