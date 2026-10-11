// Stream and event semantics that every implementation must keep:
//  - operations of one stream run in issue order (copy, kernels, copy back);
//  - several streams on independent data all finish correctly;
//  - cudaStreamWaitEvent orders work across streams;
//  - the legacy default stream waits for the other (blocking) streams, and
//    they wait for it;
//  - cudaMemcpyAsync from host memory may be reused right after the call
//    returns (the data was staged), cudaMemsetAsync, events (elapsed time,
//    query) and stream/device synchronization.
#include <vector>

#include "check.cuh"

// Slow on purpose (a few ms on the CPU): later operations that ran too early
// would see unfinished data.
__global__ void step(float* a, int n, float add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    for (int k = 0; k < 200; ++k) v = v * 1.0000001f + 1e-7f;
    a[i] = v - (v - a[i]) + add;  // a[i] + add, after the busy loop
}

__global__ void copy_add(const float* src, float* dst, int n, float add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] + add;
}

int main() {
    Checker c("streams");
    const int n = 1 << 18, nb = (n + 255) / 256, ns = 4;
    std::vector<float> host(n);

    // 1. In-order execution within each of several streams.
    cudaStream_t s[ns];
    float* buf[ns];
    std::vector<float> res[ns];
    for (int k = 0; k < ns; ++k) {
        CHECK_CUDA(cudaStreamCreate(&s[k]));
        CHECK_CUDA(cudaMalloc(&buf[k], n * sizeof(float)));
        res[k].assign(n, -1.0f);
    }
    for (int k = 0; k < ns; ++k) {
        for (int i = 0; i < n; ++i) host[i] = float(k * 1000 + i % 100);
        CHECK_CUDA(cudaMemcpyAsync(buf[k], host.data(), n * sizeof(float), cudaMemcpyHostToDevice, s[k]));
        // `host` is overwritten right away for the next stream: the copy must
        // have taken its data already.
        step<<<nb, 256, 0, s[k]>>>(buf[k], n, 1.0f);
        step<<<nb, 256, 0, s[k]>>>(buf[k], n, 2.0f);
        CHECK_CUDA(cudaMemcpyAsync(res[k].data(), buf[k], n * sizeof(float), cudaMemcpyDeviceToHost, s[k]));
    }
    for (int k = 0; k < ns; ++k) CHECK_CUDA(cudaStreamSynchronize(s[k]));
    for (int k = 0; k < ns; ++k)
        for (int i = 0; i < n; i += 997) {
            float want = float(k * 1000 + i % 100) + 3.0f;
            c.expect(res[k][i] == want, "in-order stream", k * n + i, res[k][i], want);
        }

    // 2. Cross-stream dependency through an event.
    float *a, *b;
    CHECK_CUDA(cudaMalloc(&a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b, n * sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(a, 0, n * sizeof(float), s[0]));
    step<<<nb, 256, 0, s[0]>>>(a, n, 5.0f);
    cudaEvent_t ready, t0, t1;
    CHECK_CUDA(cudaEventCreate(&ready));
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(ready, s[0]));
    CHECK_CUDA(cudaStreamWaitEvent(s[1], ready, 0));
    CHECK_CUDA(cudaEventRecord(t0, s[1]));
    copy_add<<<nb, 256, 0, s[1]>>>(a, b, n, 1.0f);  // must see a = 5
    CHECK_CUDA(cudaEventRecord(t1, s[1]));
    std::vector<float> rb(n);
    CHECK_CUDA(cudaMemcpyAsync(rb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost, s[1]));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaStreamSynchronize(s[1]));
    for (int i = 0; i < n; i += 991) c.expect(rb[i] == 6.0f, "wait event", i, rb[i], 6.0f);
    float ms = -1;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    c.expect(ms >= 0.0f, "elapsed time", -1, ms, 0);
    c.expect(cudaEventQuery(t1) == cudaSuccess, "event query after sync");

    // 3. The legacy default stream: work in s[2] issued before a default-stream
    // kernel finishes before it; s[3] work issued after it starts after it.
    CHECK_CUDA(cudaMemsetAsync(a, 0, n * sizeof(float), s[2]));
    step<<<nb, 256, 0, s[2]>>>(a, n, 7.0f);   // a = 7
    copy_add<<<nb, 256>>>(a, b, n, 1.0f);      // default stream: b = 8
    step<<<nb, 256, 0, s[3]>>>(b, n, 1.0f);   // b = 9
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(rb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i += 983) c.expect(rb[i] == 9.0f, "default stream ordering", i, rb[i], 9.0f);

    // 4. A synchronous cudaMemcpy waits for the kernels before it.
    step<<<nb, 256, 0, s[0]>>>(a, n, 1.0f);   // a = 8 (in s[0], blocking stream)
    CHECK_CUDA(cudaMemcpy(rb.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i += 977) c.expect(rb[i] == 8.0f, "cudaMemcpy after stream work", i, rb[i], 8.0f);

    for (int k = 0; k < ns; ++k) {
        CHECK_CUDA(cudaStreamQuery(s[k]) == cudaSuccess || cudaStreamQuery(s[k]) == cudaErrorNotReady
                       ? cudaSuccess
                       : cudaErrorUnknown);
        CHECK_CUDA(cudaStreamSynchronize(s[k]));
        CHECK_CUDA(cudaStreamDestroy(s[k]));
        CHECK_CUDA(cudaFree(buf[k]));
    }
    CHECK_CUDA(cudaEventDestroy(ready));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaGetLastError());
    return c.finish();
}
