// Load/store paths: __ldg, const __restrict__ pointers, vector loads and
// stores (float4, int4, double2, uchar4, int2) on aligned buffers, and the
// cache-hint intrinsics __ldcs/__stcs, __ldcg/__stcg, __ldca, __ldlu,
// __ldcv, __stwt, __stwb.
#include "stress.cuh"

__global__ void saxpy_restrict(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out,
                               float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

__global__ void ldg_gather(const int* __restrict__ idx, const float* table, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = __ldg(&idx[i]);
        float v = __ldg(table + j);
        float4 q = __ldg(reinterpret_cast<const float4*>(table) + (j >> 2));
        out[i] = v + q.x - q.w;
    }
}

__global__ void vec_float4(const float4* __restrict__ a, const float4* __restrict__ b, float4* c, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.w, x.y * y.z, x.z - y.y, x.w + y.x);
    }
}

__global__ void vec_int4_double2(const int4* a, const double2* d, int4* ai, double2* dd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int4 v = a[i];
        ai[i] = make_int4(v.w, v.z ^ v.x, v.y + 1, v.x - v.w);
        double2 q = d[i];
        dd[i] = make_double2(q.y * 2.0, q.x + q.y);
    }
}

// RGBA invert and swap channels with uchar4 loads and stores.
__global__ void vec_uchar4(const uchar4* in, uchar4* out, unsigned* sums, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uchar4 p = in[i];
        out[i] = make_uchar4(255 - p.z, 255 - p.y, 255 - p.x, p.w);
        sums[i] = unsigned(p.x) + p.y + p.z + p.w;
    }
}

__global__ void streaming_hints(const float4* in, float4* out_cs, float* out_wt, float* out_cg, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = __ldcs(in + i);  // evict-first load
        float4 r = make_float4(v.x * 2, v.y * 2, v.z * 2, v.w * 2);
        __stcs(out_cs + i, r);      // evict-first store
        const float* f = reinterpret_cast<const float*>(in);
        float a = __ldcg(f + 4 * i);   // L2 only
        float b = __ldca(f + 4 * i + 1);
        float c = __ldlu(f + 4 * i + 2);
        float d = __ldcv(f + 4 * i + 3);
        __stwt(out_wt + i, a + b + c + d);  // write-through
        __stcg(out_cg + i, a - d);
        __stwb(out_cg + n4 + i, b * c);
    }
}

__global__ void hint_scalars(const int* in, int* out, const double* din, double* dout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = __ldcg(in + i) + __ldcs(in + (i + 1) % n);
        // The int2 holding in[i] (n is even): contributes in[i] - in[i] = 0.
        int2 two = __ldg(reinterpret_cast<const int2*>(in) + i / 2);
        v += (i % 2 == 0 ? two.x : two.y) - __ldca(in + i);
        __stcs(out + i, v);
        double d = __ldcs(din + i);
        __stwt(dout + i, d * 3.0);
    }
}

int main() {
    using namespace stress;
    const int n = 1 << 20;

    {  // const __restrict__ saxpy
        auto x = random_floats(n, 1), y = random_floats(n, 2);
        Buffer<float> dx(x), dy(y), out(n);
        double k = time_device([&] { saxpy_restrict<<<n / 256, 256>>>(dx, dy, out, 1.75f, n); });
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = 1.75f * x[i] + y[i];
        });
        auto got = out.download();
        report("restrict_saxpy", close(got.data(), want.data(), n, 1e-6, 1e-7), k, ref);
    }

    {  // __ldg scalar and vector gathers
        const int tn = 1 << 16;
        auto table = random_floats(tn, 3);
        auto r = random_u32(n, 4);
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = int(r[i] % tn);
        Buffer<float> dt(table), out(n);
        Buffer<int> di(idx);
        double k = time_device([&] { ldg_gather<<<n / 256, 256>>>(di, dt, out, n); });
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                int j = idx[i], b = (j >> 2) * 4;
                want[i] = table[j] + table[b] - table[b + 3];
            }
        });
        auto got = out.download();
        report("ldg_gather", equal(got.data(), want.data(), n), k, ref);
    }

    {  // float4
        const int n4 = n / 4;
        auto a = random_floats(n, 5), b = random_floats(n, 6);
        Buffer<float> da(a), db(b), dc(n);
        double k = time_device([&] {
            vec_float4<<<n4 / 256, 256>>>(reinterpret_cast<const float4*>(da.p), reinterpret_cast<const float4*>(db.p),
                                          reinterpret_cast<float4*>(dc.p), n4);
        });
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n4; ++i) {
                const float *x = &a[4 * i], *y = &b[4 * i];
                want[4 * i] = x[0] + y[3];
                want[4 * i + 1] = x[1] * y[2];
                want[4 * i + 2] = x[2] - y[1];
                want[4 * i + 3] = x[3] + y[0];
            }
        });
        auto got = dc.download();
        report("vector_float4", equal(got.data(), want.data(), n), k, ref);
    }

    {  // int4 and double2
        const int m = n / 4;
        auto ri = random_u32(4 * size_t(m), 7);
        std::vector<int> ai(ri.begin(), ri.end());
        auto rd = random_floats(2 * size_t(m), 8);
        std::vector<double> ad(rd.begin(), rd.end());
        Buffer<int> dai(ai), oi(4 * size_t(m));
        Buffer<double> dad(ad), od(2 * size_t(m));
        double k = time_device([&] {
            vec_int4_double2<<<m / 256, 256>>>(reinterpret_cast<const int4*>(dai.p), reinterpret_cast<const double2*>(dad.p),
                                               reinterpret_cast<int4*>(oi.p), reinterpret_cast<double2*>(od.p), m);
        });
        std::vector<int> wi(4 * size_t(m));
        std::vector<double> wd(2 * size_t(m));
        for (int i = 0; i < m; ++i) {
            const int* v = &ai[4 * i];
            wi[4 * i] = v[3];
            wi[4 * i + 1] = v[2] ^ v[0];
            wi[4 * i + 2] = int(unsigned(v[1]) + 1u);
            wi[4 * i + 3] = int(unsigned(v[0]) - unsigned(v[3]));
            wd[2 * i] = ad[2 * i + 1] * 2.0;
            wd[2 * i + 1] = ad[2 * i] + ad[2 * i + 1];
        }
        auto gi = oi.download();
        auto gd = od.download();
        report("vector_int4_double2", equal(gi.data(), wi.data(), wi.size()) && equal(gd.data(), wd.data(), wd.size()), k, -1);
    }

    {  // uchar4 pixels
        auto r = random_u32(n, 9);
        Buffer<unsigned> in(r), out(n), sums(n);
        double k = time_device([&] {
            vec_uchar4<<<n / 256, 256>>>(reinterpret_cast<const uchar4*>(in.p), reinterpret_cast<uchar4*>(out.p), sums, n);
        });
        auto go = out.download();
        auto gs = sums.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            unsigned v = r[i];
            unsigned x = v & 255, y = (v >> 8) & 255, z = (v >> 16) & 255, w = v >> 24;
            unsigned want = (255 - z) | ((255 - y) << 8) | ((255 - x) << 16) | (w << 24);
            ok = go[i] == want && gs[i] == x + y + z + w;
            if (!ok) std::printf("  [%d] got %08x/%u want %08x/%u\n", i, go[i], gs[i], want, x + y + z + w);
        }
        report("vector_uchar4", ok, k, -1);
    }

    {  // cache-hint intrinsics on float4 and float
        const int n4 = n / 4;
        auto x = random_floats(n, 10);
        Buffer<float> in(x), ocs(n), owt(n4), ocg(2 * n4);
        double k = time_device([&] {
            streaming_hints<<<n4 / 256, 256>>>(reinterpret_cast<const float4*>(in.p), reinterpret_cast<float4*>(ocs.p), owt,
                                               ocg, n4);
        });
        std::vector<float> wcs(n), wwt(n4), wcg(2 * n4);
        for (int i = 0; i < n4; ++i) {
            const float* f = &x[4 * i];
            for (int c = 0; c < 4; ++c) wcs[4 * i + c] = f[c] * 2;
            wwt[i] = f[0] + f[1] + f[2] + f[3];
            wcg[i] = f[0] - f[3];
            wcg[n4 + i] = f[1] * f[2];
        }
        auto gcs = ocs.download();
        auto gwt = owt.download();
        auto gcg = ocg.download();
        bool ok = equal(gcs.data(), wcs.data(), n, "cs") && equal(gwt.data(), wwt.data(), n4, "wt") &&
                  equal(gcg.data(), wcg.data(), 2 * n4, "cg");
        report("cache_hints_float", ok, k, -1);
    }

    {  // cache hints on int and double
        auto r = random_u32(n, 11);
        std::vector<int> a(n);
        for (int i = 0; i < n; ++i) a[i] = int(r[i] % 1000000);
        auto rd = random_floats(n, 12);
        std::vector<double> d(rd.begin(), rd.end());
        Buffer<int> da(a), oa(n);
        Buffer<double> dd(d), od(n);
        double k = time_device([&] { hint_scalars<<<n / 256, 256>>>(da, oa, dd, od, n); });
        std::vector<int> wa(n);
        std::vector<double> wd(n);
        for (int i = 0; i < n; ++i) wa[i] = a[i] + a[(i + 1) % n], wd[i] = d[i] * 3.0;
        auto ga = oa.download();
        auto gd = od.download();
        report("cache_hints_int_double", equal(ga.data(), wa.data(), n) && equal(gd.data(), wd.data(), n), k, -1);
    }

    return finish();
}
