// __device__ and __constant__ globals: scalars, arrays and structs that live
// across launches, the symbol API (cudaMemcpyToSymbol / FromSymbol with
// offsets, cudaGetSymbolAddress, cudaGetSymbolSize) and a counter written by
// one kernel and read by another launched with a different block size.
#include "stress.cuh"

struct Params {
    int n;
    float scale;
    float bias;
    int mode;
    double acc;
};

constexpr int kArr = 4096;
constexpr int kCoef = 256;
constexpr int kLut = 64;

__device__ int g_scalar;
__device__ float g_array[kArr];
__device__ Params g_params;
__device__ unsigned long long g_counter;
__device__ unsigned int g_hist[16];
__device__ int g_init[8] = {3, 1, 4, 1, 5, 9, 2, 6};
__device__ Params g_init_struct = {5, 2.0f, 0.5f, 1, 0.0};
__constant__ float c_coef[kCoef];
__constant__ int c_lut[kLut];
__constant__ Params c_params;

// ---- scalar across launches -------------------------------------------------
__global__ void scalar_set(int v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) g_scalar = v;
}
__global__ void scalar_add_blocks() {
    if (threadIdx.x == 0) atomicAdd(&g_scalar, int(blockIdx.x));
}
__global__ void scalar_read(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = g_scalar + i;
}

// ---- array across launches --------------------------------------------------
__global__ void array_fill() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kArr; i += gridDim.x * blockDim.x) g_array[i] = float(i) * 0.5f;
}
__global__ void array_scale_reverse(float s) {
    // Reads the element another thread wrote in the previous launch.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kArr / 2) {
        float a = g_array[i], b = g_array[kArr - 1 - i];
        g_array[i] = b * s;
        g_array[kArr - 1 - i] = a * s;
    }
}

// ---- struct --------------------------------------------------------------------
__global__ void struct_use(const float* in, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g_params.n) {
        float v = in[i] * g_params.scale + g_params.bias;
        out[i] = g_params.mode == 1 ? v * v : v;
    }
    if (i == 0) g_params.acc = double(g_params.n) * 0.25;
}
__global__ void struct_init_read(int* out) {
    int t = threadIdx.x;
    if (t < 8) out[t] = g_init[t] * g_init_struct.n + int(g_init_struct.scale);
}

// ---- constant memory ---------------------------------------------------------
__global__ void constant_poly(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Uniform access (all lanes read the same coefficient) ...
    float v = x[i], acc = 0;
    for (int k = 7; k >= 0; --k) acc = acc * v + c_coef[k];
    // ... and divergent access (each lane a different word).
    y[i] = acc + c_coef[i % kCoef] * float(c_lut[i % kLut]) * c_params.scale;
}

// ---- symbol address ------------------------------------------------------------
__global__ void via_pointer(float* p, int n, float add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += add;
}

// ---- counter, two block sizes ------------------------------------------------
__global__ void counter_reset() {
    g_counter = 0;
    for (int i = 0; i < 16; ++i) g_hist[i] = 0;
}
__global__ void counter_count(const unsigned* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (v[i] & 7u) == 3u) {
        atomicAdd(&g_counter, 1ull);
        atomicAdd(&g_hist[v[i] >> 28], 1u);
    }
}
__global__ void counter_read(unsigned long long* out, unsigned* hist_out) {
    // Every thread of every block reads the same counter.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = g_counter + unsigned(i);
    if (i < 16) hist_out[i] = g_hist[i];
}

int main() {
    using namespace stress;

    {  // scalar written by one kernel, accumulated by another, read by a third
        const int blocks = 300, n = 1000;
        Buffer<int> out(n);
        double k = time_device([&] {
            scalar_set<<<1, 32>>>(100);
            scalar_add_blocks<<<blocks, 64>>>();
            scalar_read<<<(n + 127) / 128, 128>>>(out, n);
        });
        int host_scalar = -1;
        STRESS_CHECK(cudaMemcpyFromSymbol(&host_scalar, g_scalar, sizeof(int)));
        int want_s = 100 + blocks * (blocks - 1) / 2;
        auto got = out.download();
        bool ok = host_scalar == want_s;
        for (int i = 0; i < n && ok; ++i) ok = got[i] == want_s + i;
        report("device_scalar_across_launches", ok, k, -1);
    }

    {  // array filled by one kernel, permuted by another, read with the symbol API
        double k = time_device([&] {
            array_fill<<<8, 128>>>();
            array_scale_reverse<<<kArr / 2 / 256, 256>>>(2.0f);
        });
        std::vector<float> got(kArr), want(kArr);
        STRESS_CHECK(cudaMemcpyFromSymbol(got.data(), g_array, sizeof(float) * kArr));
        for (int i = 0; i < kArr; ++i) want[i] = float(kArr - 1 - i) * 0.5f * 2.0f;
        report("device_array_across_launches", equal(got.data(), want.data(), kArr), k, -1);
    }

    {  // struct uploaded whole, field written back by the kernel
        const int n = 5000;
        auto x = random_floats(n, 11);
        Buffer<float> in(x), out(n);
        Params p{n, 1.5f, -0.25f, 1, 0.0};
        STRESS_CHECK(cudaMemcpyToSymbol(g_params, &p, sizeof(p)));
        double k = time_device([&] { struct_use<<<(n + 255) / 256, 256>>>(in, out); });
        Params back{};
        STRESS_CHECK(cudaMemcpyFromSymbol(&back, g_params, sizeof(back)));
        auto got = out.download();
        std::vector<float> want(n);
        for (int i = 0; i < n; ++i) {
            float v = x[i] * 1.5f + -0.25f;
            want[i] = v * v;
        }
        bool ok = back.n == n && back.scale == 1.5f && back.acc == double(n) * 0.25;
        report("device_struct", ok && close(got.data(), want.data(), n, 1e-6, 1e-6), k, -1);
    }

    {  // initialized __device__ globals: no host upload
        Buffer<int> out(8);
        double k = time_device([&] { struct_init_read<<<1, 32>>>(out); });
        auto got = out.download();
        int init[8] = {3, 1, 4, 1, 5, 9, 2, 6}, want[8];
        for (int i = 0; i < 8; ++i) want[i] = init[i] * 5 + 2;
        report("device_initializers", equal(got.data(), want, 8), k, -1);
    }

    {  // __constant__ arrays and struct, uniform and divergent access
        const int n = 1 << 18;
        auto coef = random_floats(kCoef, 21);
        std::vector<int> lut(kLut);
        for (int i = 0; i < kLut; ++i) lut[i] = (i * 37) % 11 - 5;
        Params cp{0, 0.75f, 0, 0, 0};
        STRESS_CHECK(cudaMemcpyToSymbol(c_coef, coef.data(), sizeof(float) * kCoef));
        STRESS_CHECK(cudaMemcpyToSymbol(c_lut, lut.data(), sizeof(int) * kLut));
        STRESS_CHECK(cudaMemcpyToSymbol(c_params, &cp, sizeof(cp)));
        auto x = random_floats(n, 22);
        Buffer<float> dx(x), dy(n);
        double k = time_device([&] { constant_poly<<<n / 256, 256>>>(dx, dy, n); });
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                float v = x[i], acc = 0;
                for (int kk = 7; kk >= 0; --kk) acc = acc * v + coef[kk];
                want[i] = acc + coef[i % kCoef] * float(lut[i % kLut]) * 0.75f;
            }
        });
        auto got = dy.download();
        report("constant_arrays", close(got.data(), want.data(), n, 1e-5, 1e-5), k, ref);
    }

    {  // symbol copies with offsets, both directions, plus device-to-device
        std::vector<float> base(kArr);
        for (int i = 0; i < kArr; ++i) base[i] = float(i);
        STRESS_CHECK(cudaMemcpyToSymbol(g_array, base.data(), sizeof(float) * kArr));
        std::vector<float> patch(100);
        for (int i = 0; i < 100; ++i) patch[i] = -float(i) - 1;
        double t0 = now_ms();
        STRESS_CHECK(cudaMemcpyToSymbol(g_array, patch.data(), sizeof(float) * 100, sizeof(float) * 1000));
        // Device-to-device into the symbol at an odd element offset.
        Buffer<float> src(patch);
        STRESS_CHECK(cudaMemcpyToSymbol(g_array, src.p, sizeof(float) * 50, sizeof(float) * 3001, cudaMemcpyDeviceToDevice));
        std::vector<float> win(300);
        STRESS_CHECK(cudaMemcpyFromSymbol(win.data(), g_array, sizeof(float) * 300, sizeof(float) * 950));
        std::vector<float> tail(60);
        STRESS_CHECK(cudaMemcpyFromSymbol(tail.data(), g_array, sizeof(float) * 60, sizeof(float) * 2996));
        // Offsets into a __constant__ array.
        std::vector<int> lut(kLut, 0), part = {7, 8, 9};
        STRESS_CHECK(cudaMemcpyToSymbol(c_lut, lut.data(), sizeof(int) * kLut));
        STRESS_CHECK(cudaMemcpyToSymbol(c_lut, part.data(), sizeof(int) * 3, sizeof(int) * 61));
        std::vector<int> lut_back(kLut);
        STRESS_CHECK(cudaMemcpyFromSymbol(lut_back.data(), c_lut, sizeof(int) * kLut));
        double k = now_ms() - t0;
        bool ok = true;
        for (int i = 0; i < 300 && ok; ++i) {
            int g = 950 + i;
            float w = (g >= 1000 && g < 1100) ? -float(g - 1000) - 1 : float(g);
            ok = win[i] == w;
        }
        for (int i = 0; i < 60 && ok; ++i) {
            int g = 2996 + i;
            float w = (g >= 3001 && g < 3051) ? -float(g - 3001) - 1 : float(g);
            ok = tail[i] == w;
        }
        for (int i = 0; i < kLut && ok; ++i) ok = lut_back[i] == (i >= 61 ? 7 + (i - 61) : 0);
        report("symbol_offsets", ok, k, -1);
    }

    {  // cudaGetSymbolAddress / cudaGetSymbolSize
        size_t sz_arr = 0, sz_params = 0, sz_scalar = 0, sz_coef = 0;
        STRESS_CHECK(cudaGetSymbolSize(&sz_arr, g_array));
        STRESS_CHECK(cudaGetSymbolSize(&sz_params, g_params));
        STRESS_CHECK(cudaGetSymbolSize(&sz_scalar, g_scalar));
        STRESS_CHECK(cudaGetSymbolSize(&sz_coef, c_coef));
        float* p = nullptr;
        STRESS_CHECK(cudaGetSymbolAddress((void**)&p, g_array));
        std::vector<float> base(kArr);
        for (int i = 0; i < kArr; ++i) base[i] = float(i % 97);
        // Plain cudaMemcpy through the symbol address, then a kernel through it.
        STRESS_CHECK(cudaMemcpy(p, base.data(), sizeof(float) * kArr, cudaMemcpyHostToDevice));
        double t0 = now_ms();
        via_pointer<<<kArr / 128, 128>>>(p, kArr, 1.0f);
        STRESS_CHECK(cudaGetLastError());
        sync();
        double k = now_ms() - t0;
        std::vector<float> got(kArr);
        STRESS_CHECK(cudaMemcpyFromSymbol(got.data(), g_array, sizeof(float) * kArr));
        bool ok = sz_arr == sizeof(float) * kArr && sz_params == sizeof(Params) && sz_scalar == sizeof(int) &&
                  sz_coef == sizeof(float) * kCoef && p != nullptr;
        for (int i = 0; i < kArr && ok; ++i) ok = got[i] == float(i % 97) + 1.0f;
        report("symbol_address_size", ok, k, -1);
    }

    {  // counter incremented with block 128, read with block 1024 and with block 32
        const int n = 1 << 20;
        auto v = random_u32(n, 31);
        unsigned long long want = 0;
        std::vector<unsigned> want_hist(16, 0);
        for (int i = 0; i < n; ++i)
            if ((v[i] & 7u) == 3u) ++want, ++want_hist[v[i] >> 28];
        Buffer<unsigned> dv(v);
        const int readers_big = 4 * 1024, readers_small = 3 * 32;
        Buffer<unsigned long long> out_big(readers_big), out_small(readers_small);
        Buffer<unsigned> hist(16);
        double k = time_device([&] {
            counter_reset<<<1, 1>>>();
            counter_count<<<n / 128, 128>>>(dv, n);
            counter_read<<<4, 1024>>>(out_big, hist);
        });
        counter_read<<<3, 32>>>(out_small, hist);
        STRESS_CHECK(cudaGetLastError());
        sync();
        auto big = out_big.download();
        auto small = out_small.download();
        auto h = hist.download();
        unsigned long long host_counter = 0;
        STRESS_CHECK(cudaMemcpyFromSymbol(&host_counter, g_counter, sizeof(host_counter)));
        bool ok = host_counter == want;
        for (int i = 0; i < readers_big && ok; ++i) ok = big[i] == want + unsigned(i);
        for (int i = 0; i < readers_small && ok; ++i) ok = small[i] == want + unsigned(i);
        ok = ok && equal(h.data(), want_hist.data(), 16, "hist");
        report("counter_two_block_sizes", ok, k, -1);
    }

    return finish();
}
