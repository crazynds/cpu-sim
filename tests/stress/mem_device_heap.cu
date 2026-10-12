// Device heap: malloc/free in kernels (per thread, per block, across
// launches), new/delete of objects with constructors and destructors, new[]
// / delete[], and cudaDeviceSetLimit(cudaLimitMallocHeapSize).
#include "stress.cuh"

#include <new>

__device__ int* g_saved[1024];
__device__ int g_ctor_count;
__device__ int g_dtor_count;

// Per-thread scratch: each thread allocates a buffer, fills it, reduces it.
__global__ void per_thread_malloc(int* out, int* failures, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int len = 1 + i % 37;
    int* p = static_cast<int*>(malloc(len * sizeof(int)));
    if (!p) {
        atomicAdd(failures, 1);
        return;
    }
    for (int k = 0; k < len; ++k) p[k] = i + k;
    int s = 0;
    for (int k = len - 1; k >= 0; --k) s += p[k];
    out[i] = s;
    free(p);
}

// One allocation per block, shared by the block's threads through shared memory.
__global__ void per_block_malloc(const float* in, float* out, int* failures) {
    __shared__ float* buf;
    if (threadIdx.x == 0) {
        buf = static_cast<float*>(malloc(blockDim.x * sizeof(float)));
        if (!buf) atomicAdd(failures, 1);
    }
    __syncthreads();
    if (!buf) return;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    buf[threadIdx.x] = in[g];
    __syncthreads();
    out[g] = buf[blockDim.x - 1 - threadIdx.x] + buf[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) free(buf);
}

// Allocated in one launch, read and freed in a later launch.
__global__ void alloc_and_save(int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int* p = static_cast<int*>(malloc(64 * sizeof(int)));
    if (p)
        for (int k = 0; k < 64; ++k) p[k] = i * 64 + k;
    g_saved[i] = p;
}
__global__ void read_and_free(long long* sums, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int* p = g_saved[i];
    long long s = -1;
    if (p) {
        s = 0;
        for (int k = 0; k < 64; ++k) s += p[k];
        free(p);
    }
    sums[i] = s;
    g_saved[i] = nullptr;
}

struct Accum {
    int base;
    int* items;
    int count;
    __device__ Accum(int b, int c) : base(b), count(c) {
        items = new int[c];
        for (int k = 0; k < c; ++k) items[k] = b * k;
        atomicAdd(&g_ctor_count, 1);
    }
    __device__ ~Accum() {
        delete[] items;
        atomicAdd(&g_dtor_count, 1);
    }
    __device__ int total() const {
        int s = 0;
        for (int k = 0; k < count; ++k) s += items[k];
        return s + base;
    }
};

__global__ void new_delete(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Accum* a = new Accum(i, 1 + i % 9);
    out[i] = a ? a->total() : -1;
    delete a;
}

__global__ void new_array_of_doubles(double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int len = 8 + (i & 7);
    double* v = new double[len];
    for (int k = 0; k < len; ++k) v[k] = 0.5 * (i + k);
    double s = 0;
    for (int k = 0; k < len; ++k) s += v[k];
    delete[] v;
    // Placement new into a malloc'd block.
    void* raw = malloc(sizeof(double2));
    double2* d2 = new (raw) double2{s, double(len)};
    out[i] = d2->x * d2->y;
    free(raw);
}

int main() {
    using namespace stress;
    size_t heap = 0;
    STRESS_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 << 20));
    STRESS_CHECK(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize));

    {  // per-thread malloc/free
        const int n = 1 << 15;
        Buffer<int> out(n), fail(1);
        fail.zero();
        double k = time_device([&] { per_thread_malloc<<<n / 128, 128>>>(out, fail, n); });
        auto got = out.download();
        int f = fail.download()[0];
        bool ok = heap >= (64u << 20) && f == 0;
        for (int i = 0; i < n && ok; ++i) {
            int len = 1 + i % 37;
            ok = got[i] == len * i + len * (len - 1) / 2;
        }
        report("malloc_per_thread", ok, k, -1);
    }

    {  // one malloc per block, shared via __shared__ pointer
        const int n = 1 << 16, bs = 256;
        auto x = random_floats(n, 2);
        Buffer<float> in(x), out(n);
        Buffer<int> fail(1);
        fail.zero();
        double k = time_device([&] { per_block_malloc<<<n / bs, bs>>>(in, out, fail); });
        auto got = out.download();
        std::vector<float> want(n);
        for (int i = 0; i < n; ++i) {
            int b = i / bs, t = i % bs;
            want[i] = x[b * bs + bs - 1 - t] + x[i];
        }
        report("malloc_per_block_shared", fail.download()[0] == 0 && equal(got.data(), want.data(), n), k, -1);
    }

    {  // allocated in one kernel, freed in another
        const int n = 1024;
        Buffer<long long> sums(n);
        double k = time_device([&] {
            alloc_and_save<<<n / 64, 64>>>(n);
            read_and_free<<<n / 256, 256>>>(sums, n);  // different block size
        });
        auto got = sums.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) ok = got[i] == 64ll * 64 * i + 63 * 64 / 2;
        report("malloc_across_launches", ok, k, -1);
    }

    {  // new/delete of an object whose constructor uses new[] and destructor delete[]
        const int n = 10000;
        int zero = 0;
        STRESS_CHECK(cudaMemcpyToSymbol(g_ctor_count, &zero, sizeof(int)));
        STRESS_CHECK(cudaMemcpyToSymbol(g_dtor_count, &zero, sizeof(int)));
        Buffer<int> out(n);
        double t0 = now_ms();
        new_delete<<<(n + 127) / 128, 128>>>(out, n);
        STRESS_CHECK(cudaGetLastError());
        sync();
        double k = now_ms() - t0;
        int ctor = -1, dtor = -1;
        STRESS_CHECK(cudaMemcpyFromSymbol(&ctor, g_ctor_count, sizeof(int)));
        STRESS_CHECK(cudaMemcpyFromSymbol(&dtor, g_dtor_count, sizeof(int)));
        auto got = out.download();
        bool ok = ctor == n && dtor == n;
        if (!ok) std::printf("  ctor=%d dtor=%d\n", ctor, dtor);
        for (int i = 0; i < n && ok; ++i) {
            int c = 1 + i % 9;
            ok = got[i] == i * (c * (c - 1) / 2) + i;
        }
        report("new_delete_objects", ok, k, -1);
    }

    {  // new double[] / delete[] and placement new
        const int n = 8192;
        Buffer<double> out(n);
        double k = time_device([&] { new_array_of_doubles<<<n / 256, 256>>>(out, n); });
        auto got = out.download();
        std::vector<double> want(n);
        for (int i = 0; i < n; ++i) {
            int len = 8 + (i & 7);
            double s = 0;
            for (int kk = 0; kk < len; ++kk) s += 0.5 * (i + kk);
            want[i] = s * len;
        }
        report("new_array_placement", equal(got.data(), want.data(), n), k, -1);
    }

    return finish();
}
