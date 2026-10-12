// Managed (unified) memory: cudaMallocManaged used by host and device in
// turn, cudaMemPrefetchAsync, cudaMemAdvise, __managed__ variables, and a
// linked structure built by the host in managed memory and walked by a kernel.
#include "stress.cuh"

__managed__ int m_scalar = 5;
__managed__ float m_array[1024];
__managed__ unsigned long long m_sum;

__global__ void scale_add(float* p, int n, float s, float a) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = p[i] * s + a;
}

__global__ void managed_vars_kernel() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) {
        m_array[i] = m_array[i] * float(m_scalar) + 1.0f;
        atomicAdd(&m_sum, (unsigned long long)i);
    }
}

__global__ void twice(int* p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] *= 2;
}

struct Node {
    int value;
    Node* next;
};

__global__ void walk_lists(Node** heads, int lists, long long* sums) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= lists) return;
    long long s = 0;
    for (Node* p = heads[l]; p; p = p->next) {
        s += p->value;
        p->value *= 2;  // the host sees this after synchronizing
    }
    sums[l] = s;
}

#if CUDART_VERSION >= 13000
static cudaMemLocation loc_device(int dev) {
    cudaMemLocation l{};
    l.type = cudaMemLocationTypeDevice;
    l.id = dev;
    return l;
}
static cudaMemLocation loc_host() {
    cudaMemLocation l{};
    l.type = cudaMemLocationTypeHost;
    l.id = 0;
    return l;
}
#endif

static void prefetch(const void* p, size_t bytes, int dev, cudaStream_t s) {
#if CUDART_VERSION >= 13000
    STRESS_CHECK(cudaMemPrefetchAsync(p, bytes, dev == cudaCpuDeviceId ? loc_host() : loc_device(dev), 0, s));
#else
    STRESS_CHECK(cudaMemPrefetchAsync(p, bytes, dev, s));
#endif
}

static void advise(const void* p, size_t bytes, cudaMemoryAdvise a, int dev) {
#if CUDART_VERSION >= 13000
    STRESS_CHECK(cudaMemAdvise(p, bytes, a, dev == cudaCpuDeviceId ? loc_host() : loc_device(dev)));
#else
    STRESS_CHECK(cudaMemAdvise(p, bytes, a, dev));
#endif
}

int main() {
    using namespace stress;
    int concurrent = 0, managed_ok = 0;
    STRESS_CHECK(cudaDeviceGetAttribute(&managed_ok, cudaDevAttrManagedMemory, 0));
    STRESS_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0));

    {  // host and device take turns on the same allocation
        const int n = 1 << 20;
        float* p;
        STRESS_CHECK(cudaMallocManaged(&p, n * sizeof(float)));
        auto x = random_floats(n, 1);
        std::vector<float> want(x);
        double k = 0;
        for (int round = 0; round < 4; ++round) {
            for (int i = round; i < n; i += 4) p[i] = x[i] + float(round);  // host writes a quarter
            for (int i = round; i < n; i += 4) want[i] = x[i] + float(round);
            double t0 = now_ms();
            scale_add<<<256, 256>>>(p, n, 0.5f, float(round));
            STRESS_CHECK(cudaGetLastError());
            sync();
            k += now_ms() - t0;
            for (int i = 0; i < n; ++i) want[i] = want[i] * 0.5f + float(round);
        }
        bool ok = managed_ok == 1 && equal(p, want.data(), n);
        STRESS_CHECK(cudaFree(p));
        report("managed_alternating", ok, k, -1);
    }

    {  // prefetch to the device, run, prefetch back, read on the host
        const size_t n = 4 << 20;
        float* p;
        STRESS_CHECK(cudaMallocManaged(&p, n * sizeof(float)));
        auto x = random_floats(n, 2);
        std::memcpy(p, x.data(), n * sizeof(float));
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        double t0 = now_ms();
        prefetch(p, n * sizeof(float), 0, s);
        scale_add<<<512, 256, 0, s>>>(p, int(n), 2.0f, 1.0f);
        STRESS_CHECK(cudaGetLastError());
        prefetch(p, n * sizeof(float), cudaCpuDeviceId, s);
        STRESS_CHECK(cudaStreamSynchronize(s));
        double k = now_ms() - t0;
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (size_t i = 0; i < n; ++i) want[i] = x[i] * 2.0f + 1.0f;
        });
        bool ok = equal(p, want.data(), n);
        STRESS_CHECK(cudaStreamDestroy(s));
        STRESS_CHECK(cudaFree(p));
        report("managed_prefetch", ok, k, ref);
    }

    {  // cudaMemAdvise: read-mostly input, preferred location for the output
        const int n = 1 << 20;
        float *in, *out;
        STRESS_CHECK(cudaMallocManaged(&in, n * sizeof(float)));
        STRESS_CHECK(cudaMallocManaged(&out, n * sizeof(float)));
        auto x = random_floats(n, 3);
        std::memcpy(in, x.data(), n * sizeof(float));
        std::memcpy(out, x.data(), n * sizeof(float));
        advise(in, n * sizeof(float), cudaMemAdviseSetReadMostly, 0);
        advise(out, n * sizeof(float), cudaMemAdviseSetPreferredLocation, 0);
        advise(out, n * sizeof(float), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
        double k = time_device([&] { scale_add<<<256, 256>>>(out, n, 1.0f, 0.0f); });
        // Host reads the read-mostly input while the device also read it.
        double sum_in = 0;
        for (int i = 0; i < n; ++i) sum_in += in[i];
        double want_sum = 0;
        for (int i = 0; i < n; ++i) want_sum += x[i];
        bool ok = sum_in == want_sum && equal(out, x.data(), n);
        advise(in, n * sizeof(float), cudaMemAdviseUnsetReadMostly, 0);
        advise(out, n * sizeof(float), cudaMemAdviseUnsetPreferredLocation, 0);
        STRESS_CHECK(cudaFree(in));
        STRESS_CHECK(cudaFree(out));
        report("managed_advise", ok, k, -1);
    }

    {  // __managed__ variables: host initializer, host writes, kernel updates, host reads
        bool ok = m_scalar == 5;
        m_scalar = 3;
        m_sum = 0;
        for (int i = 0; i < 1024; ++i) m_array[i] = float(i) * 0.25f;
        double t0 = now_ms();
        managed_vars_kernel<<<4, 256>>>();
        STRESS_CHECK(cudaGetLastError());
        sync();
        double k = now_ms() - t0;
        for (int i = 0; i < 1024 && ok; ++i) ok = m_array[i] == float(i) * 0.25f * 3.0f + 1.0f;
        ok = ok && m_sum == 1023ull * 1024 / 2;
        // Second round with a different block size; the host changed the scalar in between.
        m_scalar = 2;
        managed_vars_kernel<<<32, 32>>>();
        STRESS_CHECK(cudaGetLastError());
        sync();
        for (int i = 0; i < 1024 && ok; ++i) ok = m_array[i] == (float(i) * 0.25f * 3.0f + 1.0f) * 2.0f + 1.0f;
        ok = ok && m_sum == 1023ull * 1024;
        report("managed_variables", ok, k, -1);
    }

    {  // linked lists built by the host in managed memory, walked and modified by a kernel
        const int lists = 256, len = 200;
        Node** heads;
        Node* pool;
        long long* sums;
        STRESS_CHECK(cudaMallocManaged(&heads, lists * sizeof(Node*)));
        STRESS_CHECK(cudaMallocManaged(&pool, size_t(lists) * len * sizeof(Node)));
        STRESS_CHECK(cudaMallocManaged(&sums, lists * sizeof(long long)));
        auto r = random_u32(size_t(lists) * len, 5);
        std::vector<long long> want(lists, 0);
        for (int l = 0; l < lists; ++l) {
            heads[l] = nullptr;
            // Nodes of a list are scattered through the pool (stride `lists`), linked back to front.
            for (int j = 0; j < len; ++j) {
                Node* nd = &pool[size_t(j) * lists + l];
                nd->value = int(r[size_t(l) * len + j] % 10000);
                nd->next = heads[l];
                heads[l] = nd;
                want[l] += nd->value;
            }
        }
        double t0 = now_ms();
        walk_lists<<<(lists + 63) / 64, 64>>>(heads, lists, sums);
        STRESS_CHECK(cudaGetLastError());
        sync();
        double k = now_ms() - t0;
        bool ok = equal(sums, want.data(), lists);
        for (int l = 0; l < lists && ok; ++l)
            for (int j = 0; j < len && ok; ++j) ok = pool[size_t(j) * lists + l].value == 2 * int(r[size_t(l) * len + j] % 10000);
        STRESS_CHECK(cudaFree(heads));
        STRESS_CHECK(cudaFree(pool));
        STRESS_CHECK(cudaFree(sums));
        report("managed_linked_lists", ok, k, -1);
    }

    {  // cudaMallocManaged with cudaMemAttachHost + cudaStreamAttachMemAsync to a stream
        const int n = 1 << 16;
        int* p;
        STRESS_CHECK(cudaMallocManaged(&p, n * sizeof(int), cudaMemAttachHost));
        for (int i = 0; i < n; ++i) p[i] = i;
        cudaStream_t s;
        STRESS_CHECK(cudaStreamCreate(&s));
        STRESS_CHECK(cudaStreamAttachMemAsync(s, p, 0, cudaMemAttachSingle));
        STRESS_CHECK(cudaStreamSynchronize(s));
        double t0 = now_ms();
        twice<<<64, 256, 0, s>>>(p, n);
        STRESS_CHECK(cudaGetLastError());
        STRESS_CHECK(cudaMemsetAsync(p, 0, 16 * sizeof(int), s));
        STRESS_CHECK(cudaStreamSynchronize(s));
        double k = now_ms() - t0;
        bool ok = concurrent >= 0;
        for (int i = 0; i < n && ok; ++i) ok = p[i] == (i < 16 ? 0 : 2 * i);
        STRESS_CHECK(cudaStreamDestroy(s));
        STRESS_CHECK(cudaFree(p));
        report("managed_attach_stream", ok, k, -1);
    }

    return finish();
}
