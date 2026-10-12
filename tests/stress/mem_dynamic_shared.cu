// Dynamic shared memory: extern arrays sized at launch, several typed arrays
// carved out of one buffer, static and dynamic together, more than 48 KiB
// through cudaFuncSetAttribute, templates over one raw extern buffer, and
// cudaFuncGetAttributes.
#include "stress.cuh"

// ---- reverse each block's segment through extern float s[] --------------------
__global__ void block_reverse(const float* in, float* out, int n) {
    extern __shared__ float s[];
    int base = blockIdx.x * blockDim.x;
    int t = threadIdx.x;
    if (base + t < n) s[t] = in[base + t];
    __syncthreads();
    int r = blockDim.x - 1 - t;
    if (base + r < n && base + t < n) out[base + t] = s[r];
}

// ---- several arrays of different types in one buffer -------------------------
// Layout: double d[nt] | int i[nt] | short h[nt] | char c[nt] | (pad to 16) | float4 v[nt/4]
__global__ void carved(const double* a, const int* b, float* out, int n) {
    extern __shared__ __align__(16) unsigned char smem[];
    int nt = blockDim.x;
    double* d = reinterpret_cast<double*>(smem);
    int* iv = reinterpret_cast<int*>(d + nt);
    short* h = reinterpret_cast<short*>(iv + nt);
    char* c = reinterpret_cast<char*>(h + nt);
    size_t off = (reinterpret_cast<size_t>(c + nt) - reinterpret_cast<size_t>(smem) + 15) & ~size_t(15);
    float4* v = reinterpret_cast<float4*>(smem + off);

    int t = threadIdx.x, g = blockIdx.x * nt + t;
    d[t] = g < n ? a[g] : 0.0;
    iv[t] = g < n ? b[g] : 0;
    h[t] = short(iv[t] & 0x7fff);
    c[t] = char(iv[t] % 100);
    if (t < nt / 4) v[t] = make_float4(float(t), float(t) * 2, float(t) * 3, float(t) * 4);
    __syncthreads();
    int u = (t + 1) % nt;  // neighbour's values
    float4 q = v[t / 4];
    float comp = (t & 3) == 0 ? q.x : (t & 3) == 1 ? q.y : (t & 3) == 2 ? q.z : q.w;
    if (g < n) out[g] = float(d[u]) + float(iv[u] >> 8) + float(h[u]) * 0.001f + float(c[u]) + comp;
}

// ---- static + dynamic ------------------------------------------------------
__global__ void static_and_dynamic(const int* in, int* out, int n) {
    __shared__ int tile[256];
    __shared__ int total;
    extern __shared__ int dyn[];  // blockDim.x ints
    int t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
    if (t == 0) total = 0;
    tile[t] = g < n ? in[g] : 0;
    __syncthreads();
    dyn[t] = tile[t] * 2 + tile[(t + 1) % blockDim.x];
    atomicAdd(&total, tile[t]);
    __syncthreads();
    if (g < n) out[g] = dyn[blockDim.x - 1 - t] + total;
}

// ---- more than 48 KiB -------------------------------------------------------
// Each block stages `per_block` floats in shared memory and writes every
// element plus the sum of the elements at stride 97 (touches the whole buffer).
__global__ void big_smem(const float* in, float* out, int per_block) {
    extern __shared__ float s[];
    const float* src = in + size_t(blockIdx.x) * per_block;
    for (int i = threadIdx.x; i < per_block; i += blockDim.x) s[i] = src[i];
    __syncthreads();
    for (int i = threadIdx.x; i < per_block; i += blockDim.x) {
        int j = per_block - 1 - i;
        out[size_t(blockIdx.x) * per_block + i] = s[j] + s[(i * 97) % per_block];
    }
}

// ---- template over a raw extern buffer --------------------------------------
template <class T>
__global__ void block_sum(const T* in, T* out, int n) {
    extern __shared__ __align__(16) unsigned char raw[];
    T* s = reinterpret_cast<T*>(raw);
    int t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
    s[t] = g < n ? in[g] : T(0);
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (t < w) s[t] += s[t + w];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

__global__ void static_only(float* out) {
    __shared__ float a[300];
    __shared__ double b[17];
    a[threadIdx.x] = float(threadIdx.x);  // blockDim.x == 300
    if (threadIdx.x < 17) b[threadIdx.x] = threadIdx.x;
    __syncthreads();
    out[threadIdx.x] = a[(threadIdx.x + 1) % 300] + float(b[threadIdx.x % 17]);
}

int main() {
    using namespace stress;

    {  // extern float s[] with sizes chosen at launch
        const int n = 1 << 20;
        auto x = random_floats(n, 1);
        Buffer<float> in(x), out(n);
        bool ok = true;
        double kbest = 0;
        for (int bs : {64, 256, 1024}) {
            double k = time_device([&] { block_reverse<<<n / bs, bs, bs * sizeof(float)>>>(in, out, n); });
            if (bs == 256) kbest = k;
            auto got = out.download();
            for (int i = 0; i < n && ok; ++i) {
                int b = i / bs, t = i % bs;
                if (got[i] != x[b * bs + bs - 1 - t]) {
                    std::printf("  bs=%d i=%d got %g\n", bs, i, got[i]);
                    ok = false;
                }
            }
        }
        report("extern_float_sizes", ok, kbest, -1);
    }

    {  // carved typed arrays
        const int n = 100000, nt = 128;
        std::vector<double> a(n);
        std::vector<int> b(n);
        auto r = random_u32(n, 7);
        for (int i = 0; i < n; ++i) a[i] = double(i) * 0.5, b[i] = int(r[i] & 0xffffff);
        Buffer<double> da(a);
        Buffer<int> db(b);
        Buffer<float> out(n);
        size_t bytes = nt * (sizeof(double) + sizeof(int) + sizeof(short) + sizeof(char));
        bytes = (bytes + 15) & ~size_t(15);
        bytes += (nt / 4) * sizeof(float4);
        int blocks = (n + nt - 1) / nt;
        double k = time_device([&] { carved<<<blocks, nt, bytes>>>(da, db, out, n); });
        std::vector<float> want(n);
        for (int g = 0; g < n; ++g) {
            int blk = g / nt, t = g % nt, u = (t + 1) % nt, gu = blk * nt + u;
            double dv = gu < n ? a[gu] : 0.0;
            int iv = gu < n ? b[gu] : 0;
            short h = short(iv & 0x7fff);
            char c = char(iv % 100);
            int q = t / 4;
            float comp = float(q) * float((t & 3) + 1);
            want[g] = float(dv) + float(iv >> 8) + float(h) * 0.001f + float(c) + comp;
        }
        auto got = out.download();
        report("carved_mixed_types", close(got.data(), want.data(), n, 1e-6, 1e-5), k, -1);
    }

    {  // static and dynamic in the same kernel
        const int n = 1 << 18, bs = 256;
        auto r = random_u32(n, 9);
        std::vector<int> x(n);
        for (int i = 0; i < n; ++i) x[i] = int(r[i] % 1000);
        Buffer<int> in(x), out(n);
        double k = time_device([&] { static_and_dynamic<<<n / bs, bs, bs * sizeof(int)>>>(in, out, n); });
        std::vector<int> want(n);
        for (int b = 0; b < n / bs; ++b) {
            int total = 0;
            for (int t = 0; t < bs; ++t) total += x[b * bs + t];
            for (int t = 0; t < bs; ++t) {
                int s = bs - 1 - t;
                want[b * bs + t] = x[b * bs + s] * 2 + x[b * bs + (s + 1) % bs] + total;
            }
        }
        auto got = out.download();
        report("static_plus_dynamic", equal(got.data(), want.data(), n), k, -1);
    }

    {  // 96 KiB of dynamic shared memory per block
        const int bytes = 96 * 1024, per_block = bytes / sizeof(float), blocks = 64;
        const size_t n = size_t(per_block) * blocks;
        STRESS_CHECK(cudaFuncSetAttribute(big_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        cudaFuncAttributes attr{};
        STRESS_CHECK(cudaFuncGetAttributes(&attr, big_smem));
        auto x = random_floats(n, 13);
        Buffer<float> in(x), out(n);
        double k = time_device([&] { big_smem<<<blocks, 512, bytes>>>(in, out, per_block); });
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < blocks; ++b)
                for (int i = 0; i < per_block; ++i) {
                    const float* s = x.data() + size_t(b) * per_block;
                    want[size_t(b) * per_block + i] = s[per_block - 1 - i] + s[(i * 97) % per_block];
                }
        });
        auto got = out.download();
        bool ok = attr.maxDynamicSharedSizeBytes == bytes;
        if (!ok) std::printf("  maxDynamicSharedSizeBytes=%d\n", attr.maxDynamicSharedSizeBytes);
        report("dynamic_96k_opt_in", ok && equal(got.data(), want.data(), n), k, ref);
    }

    {  // templated kernels sharing one raw extern buffer (int, float, double)
        const int n = 1 << 18, bs = 512, blocks = n / bs;
        auto r = random_u32(n, 15);
        std::vector<int> xi(n);
        std::vector<double> xd(n);
        for (int i = 0; i < n; ++i) xi[i] = int(r[i] % 2001) - 1000, xd[i] = double(xi[i]) * 0.25;
        Buffer<int> di(xi), oi(blocks);
        Buffer<double> dd(xd), od(blocks);
        double k = time_device([&] {
            block_sum<int><<<blocks, bs, bs * sizeof(int)>>>(di, oi, n);
            block_sum<double><<<blocks, bs, bs * sizeof(double)>>>(dd, od, n);
        });
        std::vector<int> wi(blocks, 0);
        std::vector<double> wd(blocks, 0);
        for (int i = 0; i < n; ++i) wi[i / bs] += xi[i], wd[i / bs] += xd[i];
        auto gi = oi.download();
        auto gd = od.download();
        report("template_raw_extern", equal(gi.data(), wi.data(), blocks) && equal(gd.data(), wd.data(), blocks), k, -1);
    }

    {  // cudaFuncGetAttributes fields
        cudaFuncAttributes a{}, b{}, c{};
        STRESS_CHECK(cudaFuncGetAttributes(&a, static_only));
        STRESS_CHECK(cudaFuncGetAttributes(&b, static_and_dynamic));
        STRESS_CHECK(cudaFuncGetAttributes(&c, block_reverse));
        bool ok = a.sharedSizeBytes >= 300 * sizeof(float) + 17 * sizeof(double) && a.sharedSizeBytes < 4096 &&
                  b.sharedSizeBytes >= 257 * sizeof(int) && c.sharedSizeBytes == 0 && a.numRegs > 0 && b.numRegs > 0 &&
                  a.maxThreadsPerBlock >= 256 && a.maxThreadsPerBlock <= 1024 && c.maxThreadsPerBlock == 1024 &&
                  a.ptxVersion > 0 && a.binaryVersion > 0;
        if (!ok)
            std::printf("  a.smem=%zu b.smem=%zu c.smem=%zu regs=%d,%d maxT=%d,%d ptx=%d bin=%d\n", a.sharedSizeBytes,
                        b.sharedSizeBytes, c.sharedSizeBytes, a.numRegs, b.numRegs, a.maxThreadsPerBlock,
                        c.maxThreadsPerBlock, a.ptxVersion, a.binaryVersion);
        Buffer<float> out(300);
        double k = time_device([&] { static_only<<<1, 300>>>(out); });
        auto got = out.download();
        for (int t = 0; t < 300 && ok; ++t) ok = got[t] == float((t + 1) % 300) + float(t % 17);
        report("func_get_attributes", ok, k, -1);
    }

    return finish();
}
