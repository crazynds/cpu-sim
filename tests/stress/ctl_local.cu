// Per-thread local arrays: dynamically indexed, large (1 KiB and 4 KiB per
// thread), arrays of structs, and arrays handed by pointer to device
// functions, including a recursive quicksort.
#include "stress.cuh"

__host__ __device__ inline unsigned stress_hash(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

struct Pair {
    int key;
    float val;
};

__host__ __device__ unsigned dyn_index(unsigned v) {
    int a[64];
    for (int i = 0; i < 64; ++i) a[i] = int(v * (i + 1)) ^ i;
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) r += unsigned(a[(v >> (i % 24)) & 63]) * (i + 1);
    return r;
}

// Insertion sort of 256 floats in a 1 KiB per-thread array; returns a checksum.
__host__ __device__ float sort256(unsigned seed) {
    float a[256];
    for (int i = 0; i < 256; ++i) a[i] = float(stress_hash(seed + i) & 0xffff);
    for (int i = 1; i < 256; ++i) {
        float x = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > x) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = x;
    }
    float s = 0;
    for (int i = 0; i < 256; ++i) s += a[i] * float(i & 7);
    return s;
}

__host__ __device__ int struct_array(unsigned v) {
    Pair p[16];
    for (int i = 0; i < 16; ++i) p[i] = {int(v >> i) & 255, float(i) * 0.5f};
    int best = 0;
    for (int i = 1; i < 16; ++i)
        if (p[i].key > p[best].key) best = i;
    return p[best].key * 100 + int(p[best].val * 2);
}

__host__ __device__ void fill(int* a, int n, unsigned seed) {
    for (int i = 0; i < n; ++i) a[i] = int(stress_hash(seed * 31 + i) % 1000);
}
__host__ __device__ void quicksort(int* a, int lo, int hi) {
    if (lo >= hi) return;
    int p = a[(lo + hi) / 2], i = lo, j = hi;
    while (i <= j) {
        while (a[i] < p) ++i;
        while (a[j] > p) --j;
        if (i <= j) {
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
            ++i;
            --j;
        }
    }
    quicksort(a, lo, j);
    quicksort(a, i, hi);
}
__host__ __device__ int by_pointer(unsigned seed) {
    int a[64];
    fill(a, 64, seed);
    quicksort(a, 0, 63);
    int s = 0;
    for (int i = 0; i < 64; ++i) s += a[i] * (i + 1);
    return s;
}

__host__ __device__ unsigned big_local(unsigned v) {
    unsigned a[1024];
    for (int i = 0; i < 1024; ++i) a[i] = v ^ (i * 2654435761u);
    unsigned r = 0;
    for (int i = 0; i < 1024; ++i) r += a[(i * 37 + v) & 1023] >> (i & 7);
    return r;
}

__global__ void k_dyn(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dyn_index(in[i]);
}
__global__ void k_sort(const unsigned* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sort256(in[i]);
}
__global__ void k_struct(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = struct_array(in[i]);
}
__global__ void k_pointer(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = by_pointer(in[i]);
}
__global__ void k_big(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = big_local(in[i]);
}

int main() {
    using namespace stress;
    const int n = 1 << 16;
    auto in = random_u32(n, 3);
    Buffer<unsigned> din(in), du(n);
    Buffer<float> df(n);
    Buffer<int> di(n);
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    {
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = dyn_index(in[i]);
        });
        double k = time_device([&] { k_dyn<<<n / 256, 256>>>(din, du, n); });
        report("dynamic_index_64", equal(du.download().data(), want.data(), n), k, ref);
    }
    {
        const int m = 1 << 14;
        std::vector<float> want(m);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < m; ++i) want[i] = sort256(in[i]);
        });
        double k = time_device([&] { k_sort<<<m / 128, 128>>>(din, df, m); });
        report("insertion_sort_1kib", equal(df.download().data(), want.data(), m), k, ref);
    }
    {
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = struct_array(in[i]);
        });
        double k = time_device([&] { k_struct<<<n / 256, 256>>>(din, di, n); });
        report("struct_array", equal(di.download().data(), want.data(), n), k, ref);
    }
    {
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = by_pointer(in[i]);
        });
        double k = time_device([&] { k_pointer<<<n / 128, 128>>>(din, di, n); });
        report("array_by_pointer_quicksort", equal(di.download().data(), want.data(), n), k, ref);
    }
    {
        const int m = 1 << 14;
        std::vector<unsigned> want(m);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < m; ++i) want[i] = big_local(in[i]);
        });
        double k = time_device([&] { k_big<<<m / 128, 128>>>(din, du, m); });
        report("local_4kib", equal(du.download().data(), want.data(), m), k, ref);
    }
    return finish();
}
