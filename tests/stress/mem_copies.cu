// Copies and fills: cudaMemset with non-zero bytes, device-to-device and
// cudaMemcpyDefault copies, odd sizes at odd offsets, cudaMallocPitch with a
// kernel using the pitch, cudaMemcpy2D (all directions, sub-rectangles),
// cudaMemset2D, cudaMalloc3D + cudaMemcpy3D + cudaMemset3D.
#include "stress.cuh"

__global__ void pitched_scale(float* p, size_t pitch, int w, int h, float s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        float* row = reinterpret_cast<float*>(reinterpret_cast<char*>(p) + y * pitch);
        row[x] = row[x] * s + float(y);
    }
}

__global__ void pitched_3d(cudaPitchedPtr pp, int w, int h, int d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    if (x < w && y < h && z < d) {
        char* slice = static_cast<char*>(pp.ptr) + size_t(z) * pp.pitch * h;
        int* row = reinterpret_cast<int*>(slice + size_t(y) * pp.pitch);
        row[x] = row[x] * 3 + z * 1000000 + y * 1000 + x;
    }
}

int main() {
    using namespace stress;

    {  // cudaMemset with non-zero bytes, including odd offsets and sizes
        const size_t n = 32 << 20;
        unsigned char* d;
        STRESS_CHECK(cudaMalloc(&d, n));
        double k = best_ms([&] {
            STRESS_CHECK(cudaMemset(d, 0xAB, n));
            STRESS_CHECK(cudaMemset(d + 1, 0x01, 3));
            STRESS_CHECK(cudaMemset(d + 1001, 0x7F, 7777));
            STRESS_CHECK(cudaMemset(d + n - 5, 0xFF, 5));
            sync();
        });
        std::vector<unsigned char> want(n), got(n);
        double ref = best_ms([&] {
            std::memset(want.data(), 0xAB, n);
            std::memset(want.data() + 1, 0x01, 3);
            std::memset(want.data() + 1001, 0x7F, 7777);
            std::memset(want.data() + n - 5, 0xFF, 5);
        });
        STRESS_CHECK(cudaMemcpy(got.data(), d, n, cudaMemcpyDeviceToHost));
        // An int buffer filled with byte 0x01 reads back as 0x01010101.
        int* di;
        STRESS_CHECK(cudaMalloc(&di, 1000 * sizeof(int)));
        STRESS_CHECK(cudaMemset(di, 0x01, 1000 * sizeof(int)));
        std::vector<int> gi(1000);
        STRESS_CHECK(cudaMemcpy(gi.data(), di, 1000 * sizeof(int), cudaMemcpyDeviceToHost));
        bool ok = equal(got.data(), want.data(), n, "bytes");
        for (int i = 0; i < 1000 && ok; ++i) ok = gi[i] == 0x01010101;
        STRESS_CHECK(cudaFree(d));
        STRESS_CHECK(cudaFree(di));
        report("memset_nonzero", ok, k, ref);
    }

    {  // large device-to-device copy
        const size_t n = 16 << 20;  // floats, 64 MiB
        auto x = random_floats(n, 3);
        Buffer<float> a(x), b(n);
        double k = best_ms([&] {
            STRESS_CHECK(cudaMemcpy(b.p, a.p, n * sizeof(float), cudaMemcpyDeviceToDevice));
            sync();
        });
        std::vector<float> host(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (size_t i = 0; i < n; ++i) host[i] = x[i];
        });
        auto got = b.download();
        report("memcpy_d2d_64mib", equal(got.data(), x.data(), n), k, ref);
    }

    {  // cudaMemcpyDefault in every direction (the runtime infers it from UVA)
        const size_t n = 1 << 20;
        auto x = random_u32(n, 5);
        unsigned *a, *b;
        STRESS_CHECK(cudaMalloc(&a, n * 4));
        STRESS_CHECK(cudaMalloc(&b, n * 4));
        std::vector<unsigned> back(n, 0);
        double k = best_ms([&] {
            STRESS_CHECK(cudaMemcpy(a, x.data(), n * 4, cudaMemcpyDefault));
            STRESS_CHECK(cudaMemcpy(b, a, n * 4, cudaMemcpyDefault));
            STRESS_CHECK(cudaMemcpy(back.data(), b, n * 4, cudaMemcpyDefault));
        });
        // Host to host through the runtime too.
        std::vector<unsigned> hh(n, 0);
        STRESS_CHECK(cudaMemcpy(hh.data(), back.data(), n * 4, cudaMemcpyDefault));
        STRESS_CHECK(cudaMemcpy(hh.data(), hh.data() + 1, 4, cudaMemcpyHostToHost));
        bool ok = equal(back.data(), x.data(), n) && hh[0] == x[1] && hh[1] == x[1] && hh[n - 1] == x[n - 1];
        STRESS_CHECK(cudaFree(a));
        STRESS_CHECK(cudaFree(b));
        report("memcpy_default_kind", ok, k, -1);
    }

    {  // odd sizes at odd offsets, neighbours untouched
        const size_t cap = 1 << 16;
        unsigned char *d1, *d2;
        STRESS_CHECK(cudaMalloc(&d1, cap));
        STRESS_CHECK(cudaMalloc(&d2, cap));
        auto src32 = random_u32(cap / 4, 17);
        const unsigned char* src = reinterpret_cast<const unsigned char*>(src32.data());
        bool ok = true;
        double t0 = now_ms();
        for (size_t size : {1, 3, 7, 13, 61, 255, 4097}) {
            for (size_t off1 : {0, 1, 3, 7}) {
                for (size_t off2 : {1, 3, 7, 0}) {
                    STRESS_CHECK(cudaMemset(d1, 0x5A, cap));
                    STRESS_CHECK(cudaMemset(d2, 0xC3, cap));
                    STRESS_CHECK(cudaMemcpy(d1 + off1 + 100, src + off2, size, cudaMemcpyHostToDevice));
                    STRESS_CHECK(cudaMemcpy(d2 + off2 + 9, d1 + off1 + 100, size, cudaMemcpyDeviceToDevice));
                    std::vector<unsigned char> g1(size + 2 * 16), g2(size + 2 * 16);
                    STRESS_CHECK(cudaMemcpy(g1.data(), d1 + off1 + 100 - 16, size + 32, cudaMemcpyDeviceToHost));
                    STRESS_CHECK(cudaMemcpy(g2.data(), d2 + off2 + 9 - 9, size + 32, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < size + 32 && ok; ++i) {
                        unsigned char w1 = (i >= 16 && i < 16 + size) ? src[off2 + i - 16] : 0x5A;
                        unsigned char w2 = (i >= 9 && i < 9 + size) ? src[off2 + i - 9] : 0xC3;
                        if (g1[i] != w1 || g2[i] != w2) {
                            std::printf("  size=%zu off1=%zu off2=%zu i=%zu got %02x/%02x want %02x/%02x\n", size, off1,
                                        off2, i, g1[i], g2[i], w1, w2);
                            ok = false;
                        }
                    }
                }
            }
        }
        double k = now_ms() - t0;
        STRESS_CHECK(cudaFree(d1));
        STRESS_CHECK(cudaFree(d2));
        report("odd_sizes_offsets", ok, k, -1);
    }

    {  // cudaMallocPitch + cudaMemcpy2D in + kernel on the pitch + cudaMemcpy2D out with a padded host pitch
        const int w = 1000, h = 300;
        float* d;
        size_t pitch = 0;
        STRESS_CHECK(cudaMallocPitch(&d, &pitch, w * sizeof(float), h));
        auto x = random_floats(size_t(w) * h, 19);
        STRESS_CHECK(cudaMemcpy2D(d, pitch, x.data(), w * sizeof(float), w * sizeof(float), h, cudaMemcpyHostToDevice));
        dim3 bs(32, 8), gs((w + 31) / 32, (h + 7) / 8);
        double t0 = now_ms();
        pitched_scale<<<gs, bs>>>(d, pitch, w, h, 2.0f);
        STRESS_CHECK(cudaGetLastError());
        sync();
        double k = now_ms() - t0;
        const int hp = w + 24;  // host pitch in floats
        std::vector<float> got(size_t(hp) * h, -7.0f);
        STRESS_CHECK(cudaMemcpy2D(got.data(), hp * sizeof(float), d, pitch, w * sizeof(float), h, cudaMemcpyDeviceToHost));
        bool ok = pitch >= w * sizeof(float) && pitch % 128 == 0;
        if (!ok) std::printf("  pitch=%zu\n", pitch);
        for (int y = 0; y < h && ok; ++y)
            for (int xx = 0; xx < hp && ok; ++xx) {
                float want = xx < w ? x[size_t(y) * w + xx] * 2.0f + float(y) : -7.0f;
                if (got[size_t(y) * hp + xx] != want) {
                    std::printf("  (%d,%d) got %g want %g\n", xx, y, got[size_t(y) * hp + xx], want);
                    ok = false;
                }
            }
        STRESS_CHECK(cudaFree(d));
        report("malloc_pitch_kernel", ok, k, -1);
    }

    {  // cudaMemset2D on a sub-rectangle and cudaMemcpy2D device-to-device between two pitched buffers
        const int w = 777, h = 129;  // bytes, rows
        unsigned char *a, *b;
        size_t pa, pb;
        STRESS_CHECK(cudaMallocPitch(&a, &pa, w, h));
        STRESS_CHECK(cudaMallocPitch(&b, &pb, w + 300, h + 5));
        double t0 = now_ms();
        STRESS_CHECK(cudaMemset2D(a, pa, 0x11, w, h));
        STRESS_CHECK(cudaMemset2D(a + 5 * pa + 13, pa, 0xEE, 101, 33));  // rows 5..37, cols 13..113
        STRESS_CHECK(cudaMemset2D(b, pb, 0x00, w + 300, h + 5));
        // Copy a 211 x 50 window of a starting at (7, 3) into b at (299, 2).
        STRESS_CHECK(cudaMemcpy2D(b + 2 * pb + 299, pb, a + 3 * pa + 7, pa, 211, 50, cudaMemcpyDeviceToDevice));
        sync();
        double k = now_ms() - t0;
        std::vector<unsigned char> ga(size_t(w) * h), gb(size_t(w + 300) * (h + 5));
        STRESS_CHECK(cudaMemcpy2D(ga.data(), w, a, pa, w, h, cudaMemcpyDeviceToHost));
        STRESS_CHECK(cudaMemcpy2D(gb.data(), w + 300, b, pb, w + 300, h + 5, cudaMemcpyDeviceToHost));
        auto wa = [&](int x, int y) -> unsigned char {
            return (y >= 5 && y < 38 && x >= 13 && x < 114) ? 0xEE : 0x11;
        };
        bool ok = true;
        for (int y = 0; y < h && ok; ++y)
            for (int x = 0; x < w && ok; ++x) ok = ga[size_t(y) * w + x] == wa(x, y);
        for (int y = 0; y < h + 5 && ok; ++y)
            for (int x = 0; x < w + 300 && ok; ++x) {
                unsigned char want = (y >= 2 && y < 52 && x >= 299 && x < 510) ? wa(x - 299 + 7, y - 2 + 3) : 0;
                if (gb[size_t(y) * (w + 300) + x] != want) {
                    std::printf("  b(%d,%d) got %02x want %02x\n", x, y, gb[size_t(y) * (w + 300) + x], want);
                    ok = false;
                }
            }
        STRESS_CHECK(cudaFree(a));
        STRESS_CHECK(cudaFree(b));
        report("memset2d_memcpy2d_d2d", ok, k, -1);
    }

    {  // cudaMemcpy2D host to host and host-to-device of a column strip
        const int w = 64, h = 40;
        std::vector<int> src(w * h), dst(w * h, 0);
        for (int i = 0; i < w * h; ++i) src[i] = i;
        // Column strip: 3 ints wide, every row, host to host.
        STRESS_CHECK(cudaMemcpy2D(dst.data() + 10, w * sizeof(int), src.data() + 20, w * sizeof(int), 3 * sizeof(int), h,
                                  cudaMemcpyHostToHost));
        int* d;
        size_t p;
        STRESS_CHECK(cudaMallocPitch(&d, &p, 5 * sizeof(int), h));
        double t0 = now_ms();
        STRESS_CHECK(cudaMemcpy2D(d, p, src.data() + 1, w * sizeof(int), 5 * sizeof(int), h, cudaMemcpyHostToDevice));
        std::vector<int> col(5 * h);
        STRESS_CHECK(cudaMemcpy2D(col.data(), 5 * sizeof(int), d, p, 5 * sizeof(int), h, cudaMemcpyDeviceToHost));
        double k = now_ms() - t0;
        bool ok = true;
        for (int y = 0; y < h && ok; ++y)
            for (int x = 0; x < w && ok; ++x) ok = dst[y * w + x] == ((x >= 10 && x < 13) ? y * w + x + 10 : 0);
        for (int y = 0; y < h && ok; ++y)
            for (int x = 0; x < 5 && ok; ++x) ok = col[y * 5 + x] == y * w + x + 1;
        STRESS_CHECK(cudaFree(d));
        report("memcpy2d_strips", ok, k, -1);
    }

    {  // cudaMalloc3D + cudaMemset3D + cudaMemcpy3D in, kernel, cudaMemcpy3D out
        const int w = 100, h = 37, dd = 11;
        cudaExtent ext = make_cudaExtent(w * sizeof(int), h, dd);
        cudaPitchedPtr dp;
        STRESS_CHECK(cudaMalloc3D(&dp, ext));
        STRESS_CHECK(cudaMemset3D(dp, 0, ext));
        std::vector<int> host(size_t(w) * h * dd);
        for (size_t i = 0; i < host.size(); ++i) host[i] = int(i % 1000);
        cudaMemcpy3DParms in{};
        in.srcPtr = make_cudaPitchedPtr(host.data(), w * sizeof(int), w, h);
        in.dstPtr = dp;
        in.extent = ext;
        in.kind = cudaMemcpyHostToDevice;
        // Copy everything except the first slice; the first slice stays zero (memset).
        in.srcPos = make_cudaPos(0, 0, 1);
        in.dstPos = make_cudaPos(0, 0, 1);
        in.extent = make_cudaExtent(w * sizeof(int), h, dd - 1);
        STRESS_CHECK(cudaMemcpy3D(&in));
        double t0 = now_ms();
        pitched_3d<<<dim3((w + 63) / 64, h, dd), 64>>>(dp, w, h, dd);
        STRESS_CHECK(cudaGetLastError());
        sync();
        double k = now_ms() - t0;
        std::vector<int> got(host.size(), -1);
        cudaMemcpy3DParms out{};
        out.srcPtr = dp;
        out.dstPtr = make_cudaPitchedPtr(got.data(), w * sizeof(int), w, h);
        out.extent = ext;
        out.kind = cudaMemcpyDeviceToHost;
        STRESS_CHECK(cudaMemcpy3D(&out));
        bool ok = dp.pitch >= w * sizeof(int) && dp.xsize == w * sizeof(int) && dp.ysize == size_t(h);
        if (!ok) std::printf("  pitch=%zu xsize=%zu ysize=%zu\n", dp.pitch, dp.xsize, dp.ysize);
        for (int z = 0; z < dd && ok; ++z)
            for (int y = 0; y < h && ok; ++y)
                for (int x = 0; x < w && ok; ++x) {
                    size_t i = (size_t(z) * h + y) * w + x;
                    int base = z == 0 ? 0 : host[i];
                    int want = base * 3 + z * 1000000 + y * 1000 + x;
                    if (got[i] != want) {
                        std::printf("  (%d,%d,%d) got %d want %d\n", x, y, z, got[i], want);
                        ok = false;
                    }
                }
        STRESS_CHECK(cudaFree(dp.ptr));
        report("malloc3d_memcpy3d", ok, k, -1);
    }

    return finish();
}
