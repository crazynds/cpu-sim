// CUDA Graphs: stream capture of copies and kernels replayed many times,
// fork/join capture across streams with events, a graph built explicitly with
// kernel / memset / memcpy nodes, and updating a kernel node's parameters in
// an instantiated graph.
#include "stress.cuh"

__global__ void axpb(const float* x, float* y, int n, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}
__global__ void add_arrays(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
__global__ void accumulate(int* acc, const int* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += v[i];
}
__global__ void set_value(int* p, int n, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

int main() {
    using namespace stress;

    {  // capture H2D, two kernels, D2H; replay with new host input each time
        const int n = 1 << 18, reps = 20;
        float *h_in, *h_out, *d_x, *d_y;
        STRESS_CHECK(cudaMallocHost(&h_in, n * sizeof(float)));
        STRESS_CHECK(cudaMallocHost(&h_out, n * sizeof(float)));
        STRESS_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        STRESS_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        cudaStreamCaptureStatus st0, st1;
        STRESS_CHECK(cudaStreamIsCapturing(s, &st0));
        STRESS_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        STRESS_CHECK(cudaStreamIsCapturing(s, &st1));
        STRESS_CHECK(cudaMemcpyAsync(d_x, h_in, n * sizeof(float), cudaMemcpyHostToDevice, s));
        axpb<<<n / 256, 256, 0, s>>>(d_x, d_y, n, 2.0f, 1.0f);
        add_arrays<<<n / 256, 256, 0, s>>>(d_x, d_y, d_y, n);
        STRESS_CHECK(cudaMemcpyAsync(h_out, d_y, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        cudaGraph_t g;
        STRESS_CHECK(cudaStreamEndCapture(s, &g));
        size_t nodes = 0;
        STRESS_CHECK(cudaGraphGetNodes(g, nullptr, &nodes));
        cudaGraphExec_t ge;
        STRESS_CHECK(cudaGraphInstantiate(&ge, g, 0));
        bool ok = st0 == cudaStreamCaptureStatusNone && st1 == cudaStreamCaptureStatusActive && nodes == 4;
        if (!ok) std::printf("  st0=%d st1=%d nodes=%zu\n", int(st0), int(st1), nodes);
        double t0 = now_ms();
        for (int r = 0; r < reps && ok; ++r) {
            for (int i = 0; i < n; ++i) h_in[i] = float(i % 1000) * 0.5f + float(r);
            STRESS_CHECK(cudaGraphLaunch(ge, s));
            STRESS_CHECK(cudaStreamSynchronize(s));
            for (int i = 0; i < n && ok; ++i) {
                float x = float(i % 1000) * 0.5f + float(r);
                ok = h_out[i] == x + (2.0f * x + 1.0f);
            }
        }
        double k = (now_ms() - t0) / reps;
        STRESS_CHECK(cudaGraphExecDestroy(ge));
        STRESS_CHECK(cudaGraphDestroy(g));
        STRESS_CHECK(cudaStreamDestroy(s));
        STRESS_CHECK(cudaFreeHost(h_in));
        STRESS_CHECK(cudaFreeHost(h_out));
        STRESS_CHECK(cudaFree(d_x));
        STRESS_CHECK(cudaFree(d_y));
        report("capture_replay", ok, k, -1);
    }

    {  // fork/join capture: two branches in two streams, joined by events
        const int n = 1 << 16;
        Buffer<float> x(random_floats(n, 3)), a(n), b(n), c(n);
        cudaStream_t s0, s1;
        STRESS_CHECK(cudaStreamCreate(&s0));
        STRESS_CHECK(cudaStreamCreate(&s1));
        cudaEvent_t fork, join;
        STRESS_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        STRESS_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
        STRESS_CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeThreadLocal));
        STRESS_CHECK(cudaEventRecord(fork, s0));
        STRESS_CHECK(cudaStreamWaitEvent(s1, fork, 0));
        axpb<<<n / 256, 256, 0, s0>>>(x, a, n, 3.0f, 0.0f);
        axpb<<<n / 256, 256, 0, s1>>>(x, b, n, -1.0f, 4.0f);
        STRESS_CHECK(cudaEventRecord(join, s1));
        STRESS_CHECK(cudaStreamWaitEvent(s0, join, 0));
        add_arrays<<<n / 256, 256, 0, s0>>>(a, b, c, n);
        cudaGraph_t g;
        STRESS_CHECK(cudaStreamEndCapture(s0, &g));
        size_t nodes = 0;
        STRESS_CHECK(cudaGraphGetNodes(g, nullptr, &nodes));
        cudaGraphExec_t ge;
        STRESS_CHECK(cudaGraphInstantiate(&ge, g, 0));
        double k = time_device([&] { STRESS_CHECK(cudaGraphLaunch(ge, s0)); });
        auto hx = x.download();
        auto got = c.download();
        std::vector<float> want(n);
        for (int i = 0; i < n; ++i) want[i] = 3.0f * hx[i] + (-1.0f * hx[i] + 4.0f);
        bool ok = nodes == 3 && equal(got.data(), want.data(), n);
        if (nodes != 3) std::printf("  nodes=%zu\n", nodes);
        STRESS_CHECK(cudaGraphExecDestroy(ge));
        STRESS_CHECK(cudaGraphDestroy(g));
        STRESS_CHECK(cudaEventDestroy(fork));
        STRESS_CHECK(cudaEventDestroy(join));
        STRESS_CHECK(cudaStreamDestroy(s0));
        STRESS_CHECK(cudaStreamDestroy(s1));
        report("capture_fork_join", ok, k, -1);
    }

    {  // explicit graph: memset -> kernel -> kernel -> memcpy D2H, launched several times
        const int n = 4096;
        Buffer<int> acc(n), v(n);
        std::vector<int> host(n, -1);
        cudaGraph_t g;
        STRESS_CHECK(cudaGraphCreate(&g, 0));

        cudaMemsetParams mp{};
        mp.dst = acc.p;
        mp.value = 0;
        mp.elementSize = sizeof(int);
        mp.width = n;
        mp.height = 1;
        mp.pitch = 0;
        cudaGraphNode_t n_memset, n_set, n_acc, n_copy;
        STRESS_CHECK(cudaGraphAddMemsetNode(&n_memset, g, nullptr, 0, &mp));

        int* vp = v.p;
        int nn = n, val = 7;
        void* set_args[] = {&vp, &nn, &val};
        cudaKernelNodeParams kp{};
        kp.func = reinterpret_cast<void*>(set_value);
        kp.gridDim = dim3(n / 128);
        kp.blockDim = dim3(128);
        kp.sharedMemBytes = 0;
        kp.kernelParams = set_args;
        kp.extra = nullptr;
        STRESS_CHECK(cudaGraphAddKernelNode(&n_set, g, nullptr, 0, &kp));

        int* ap = acc.p;
        void* acc_args[] = {&ap, &vp, &nn};
        cudaKernelNodeParams kp2 = kp;
        kp2.func = reinterpret_cast<void*>(accumulate);
        kp2.kernelParams = acc_args;
        cudaGraphNode_t deps[] = {n_memset, n_set};
        STRESS_CHECK(cudaGraphAddKernelNode(&n_acc, g, deps, 2, &kp2));
        STRESS_CHECK(cudaGraphAddMemcpyNode1D(&n_copy, g, &n_acc, 1, host.data(), acc.p, n * sizeof(int),
                                              cudaMemcpyDeviceToHost));
        cudaGraphExec_t ge;
        STRESS_CHECK(cudaGraphInstantiate(&ge, g, 0));
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        double t0 = now_ms();
        bool ok = true;
        for (int r = 0; r < 3 && ok; ++r) {
            std::fill(host.begin(), host.end(), -1);
            STRESS_CHECK(cudaGraphLaunch(ge, s));
            STRESS_CHECK(cudaStreamSynchronize(s));
            for (int i = 0; i < n && ok; ++i) ok = host[i] == 7;  // memset to 0 each launch, then + 7
        }
        double k = (now_ms() - t0) / 3;

        // Update the set_value node's argument in the executable graph.
        int val2 = 31;
        void* set_args2[] = {&vp, &nn, &val2};
        cudaKernelNodeParams kp3 = kp;
        kp3.kernelParams = set_args2;
        STRESS_CHECK(cudaGraphExecKernelNodeSetParams(ge, n_set, &kp3));
        STRESS_CHECK(cudaGraphLaunch(ge, s));
        STRESS_CHECK(cudaStreamSynchronize(s));
        for (int i = 0; i < n && ok; ++i) ok = host[i] == 31;
        size_t nodes = 0;
        STRESS_CHECK(cudaGraphGetNodes(g, nullptr, &nodes));
        ok = ok && nodes == 4;
        STRESS_CHECK(cudaGraphExecDestroy(ge));
        STRESS_CHECK(cudaGraphDestroy(g));
        STRESS_CHECK(cudaStreamDestroy(s));
        report("explicit_graph_nodes", ok, k, -1);
    }

    {  // a captured graph of many small kernels (a chain of 64), launched repeatedly
        const int n = 1024, chain = 64;
        Buffer<int> acc(n), one(n);
        STRESS_CHECK(cudaMemset(acc.p, 0, n * sizeof(int)));
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        set_value<<<n / 256, 256, 0, s>>>(one, n, 1);
        STRESS_CHECK(cudaStreamSynchronize(s));
        STRESS_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < chain; ++i) accumulate<<<n / 256, 256, 0, s>>>(acc, one, n);
        cudaGraph_t g;
        STRESS_CHECK(cudaStreamEndCapture(s, &g));
        cudaGraphExec_t ge;
        STRESS_CHECK(cudaGraphInstantiateWithFlags(&ge, g, 0));
        STRESS_CHECK(cudaGraphUpload(ge, s));
        const int launches = 5;
        double t0 = now_ms();
        for (int r = 0; r < launches; ++r) STRESS_CHECK(cudaGraphLaunch(ge, s));
        STRESS_CHECK(cudaStreamSynchronize(s));
        double k = (now_ms() - t0) / launches;
        auto got = acc.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) ok = got[i] == chain * launches;
        STRESS_CHECK(cudaGraphExecDestroy(ge));
        STRESS_CHECK(cudaGraphDestroy(g));
        STRESS_CHECK(cudaStreamDestroy(s));
        report("captured_kernel_chain", ok, k, -1);
    }

    return finish();
}
