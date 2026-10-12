// Physics, finance and simulation workloads: an all-pairs n-body step with
// shared-memory tiling (GPU Gems 3, ch. 31), Black-Scholes call and put
// pricing (as in the CUDA sample), Monte Carlo estimation of pi with a
// per-thread xorshift generator and block reduction plus atomicAdd, and a
// Mandelbrot set with per-pixel iteration counts.
#include "stress.cuh"

// ---------------------------------------------------------------------------
// N-body: one leapfrog-style step. pos.w holds the mass.

#define NBODY_TILE 256
#define SOFTENING2 0.01f

__device__ __forceinline__ float3 body_body(float4 bi, float4 bj, float3 ai) {
    float3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    float dist2 = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING2;
    float inv = rsqrtf(dist2);
    float inv3 = inv * inv * inv;
    float s = bj.w * inv3;
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__global__ void nbody_step(const float4* __restrict__ pos, const float4* __restrict__ vel, float4* __restrict__ new_pos,
                           float4* __restrict__ new_vel, float3* __restrict__ acc_out, int n, float dt) {
    __shared__ float4 tile[NBODY_TILE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 me = i < n ? pos[i] : make_float4(0, 0, 0, 0);
    float3 a = make_float3(0, 0, 0);
    for (int base = 0; base < n; base += NBODY_TILE) {
        int j = base + threadIdx.x;
        tile[threadIdx.x] = j < n ? pos[j] : make_float4(0, 0, 0, 0);
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < NBODY_TILE; ++k) a = body_body(me, tile[k], a);
        __syncthreads();
    }
    if (i < n) {
        float4 v = vel[i];
        v.x += a.x * dt;
        v.y += a.y * dt;
        v.z += a.z * dt;
        me.x += v.x * dt;
        me.y += v.y * dt;
        me.z += v.z * dt;
        new_vel[i] = v;
        new_pos[i] = me;
        acc_out[i] = a;
    }
}

// ---------------------------------------------------------------------------
// Black-Scholes (CUDA sample BlackScholes).

__device__ inline float cnd_dev(float d) {
    const float A1 = 0.31938153f, A2 = -0.356563782f, A3 = 1.781477937f, A4 = -1.821255978f, A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;
    float K = __fdividef(1.0f, 1.0f + 0.2316419f * fabsf(d));
    float c = RSQRT2PI * __expf(-0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0) c = 1.0f - c;
    return c;
}

__global__ void black_scholes(float* __restrict__ call, float* __restrict__ put, const float* __restrict__ S,
                              const float* __restrict__ X, const float* __restrict__ T, float R, float V, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float s = S[i], x = X[i], t = T[i];
        float sqrtT = sqrtf(t);
        float d1 = __fdividef(__logf(s / x) + (R + 0.5f * V * V) * t, V * sqrtT);
        float d2 = d1 - V * sqrtT;
        float cnd1 = cnd_dev(d1), cnd2 = cnd_dev(d2);
        float expRT = __expf(-R * t);
        call[i] = s * cnd1 - x * expRT * cnd2;
        put[i] = x * expRT * (1.0f - cnd2) - s * (1.0f - cnd1);
    }
}

static inline float cnd_host(float d) {
    const float A1 = 0.31938153f, A2 = -0.356563782f, A3 = 1.781477937f, A4 = -1.821255978f, A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;
    float K = 1.0f / (1.0f + 0.2316419f * std::fabs(d));
    float c = RSQRT2PI * std::exp(-0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    return d > 0 ? 1.0f - c : c;
}

// ---------------------------------------------------------------------------
// Monte Carlo pi: xorshift32 per thread, integer hit test (exact), warp
// shuffle + shared-memory block reduction, one atomicAdd per block.

__host__ __device__ inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__host__ __device__ inline uint32_t mc_seed(uint32_t tid) {
    uint32_t x = tid * 0x9E3779B9u + 0x7F4A7C15u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x ? x : 1u;
}

template <int SAMPLES>
__host__ __device__ inline uint32_t mc_thread_hits(uint32_t tid) {
    uint32_t s = mc_seed(tid), hits = 0;
    for (int k = 0; k < SAMPLES; ++k) {
        uint64_t x = xorshift32(s) >> 8, y = xorshift32(s) >> 8;  // 24-bit coordinates in [0, 2^24)
        hits += (x * x + y * y < (1ull << 48)) ? 1u : 0u;
    }
    return hits;
}

template <int SAMPLES, int BLOCK>
__global__ void monte_carlo_pi(unsigned long long* total) {
    __shared__ uint32_t warp_sums[BLOCK / 32];
    uint32_t tid = blockIdx.x * BLOCK + threadIdx.x;
    uint32_t h = mc_thread_hits<SAMPLES>(tid);
    for (int off = 16; off > 0; off >>= 1) h += __shfl_xor_sync(0xffffffffu, h, off);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = h;
    __syncthreads();
    if (warp == 0) {
        h = lane < BLOCK / 32 ? warp_sums[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) h += __shfl_down_sync(0xffffffffu, h, off);
        if (lane == 0) atomicAdd(total, (unsigned long long)h);
    }
}

// ---------------------------------------------------------------------------
// Mandelbrot.

__global__ void mandelbrot(int* __restrict__ iters, int w, int h, float x0, float y0, float dx, float dy, int max_iter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    float cx = x0 + px * dx, cy = y0 + py * dy;
    float x = 0, y = 0;
    int it = 0;
    while (x * x + y * y <= 4.0f && it < max_iter) {
        float xt = x * x - y * y + cx;
        y = 2.0f * x * y + cy;
        x = xt;
        ++it;
    }
    iters[py * w + px] = it;
}

// ---------------------------------------------------------------------------

int main() {
    using namespace stress;

    // N-body.
    for (int n : {4096, 16384}) {
        const float dt = 0.001f;
        auto rp = random_floats(4 * n, 11, -10.0f, 10.0f);
        auto rv = random_floats(4 * n, 12, -1.0f, 1.0f);
        std::vector<float4> pos(n), vel(n);
        for (int i = 0; i < n; ++i) {
            pos[i] = make_float4(rp[4 * i], rp[4 * i + 1], rp[4 * i + 2], 0.5f + 0.25f * (rp[4 * i + 3] + 10.0f) / 20.0f);
            vel[i] = make_float4(rv[4 * i], rv[4 * i + 1], rv[4 * i + 2], 0.0f);
        }
        std::vector<float4> want_pos(n), want_vel(n);
        std::vector<float> want_acc(3 * n), acc_scale(n);
        double ref = best_ms([&] {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                float px = pos[i].x, py = pos[i].y, pz = pos[i].z;
                float ax = 0, ay = 0, az = 0, mag = 0;
#pragma omp simd reduction(+ : ax, ay, az, mag)
                for (int j = 0; j < n; ++j) {
                    float rx = pos[j].x - px, ry = pos[j].y - py, rz = pos[j].z - pz;
                    float d2 = rx * rx + ry * ry + rz * rz + SOFTENING2;
                    float inv = 1.0f / std::sqrt(d2);
                    float s = pos[j].w * inv * inv * inv;
                    ax += rx * s;
                    ay += ry * s;
                    az += rz * s;
                    mag += s * std::sqrt(d2);
                }
                want_acc[3 * i] = ax;
                want_acc[3 * i + 1] = ay;
                want_acc[3 * i + 2] = az;
                acc_scale[i] = mag;
                float4 v = vel[i];
                v.x += ax * dt;
                v.y += ay * dt;
                v.z += az * dt;
                want_vel[i] = v;
                want_pos[i] = make_float4(px + v.x * dt, py + v.y * dt, pz + v.z * dt, pos[i].w);
            }
        });
        Buffer<float4> dpos(pos), dvel(vel), dnp(n), dnv(n);
        Buffer<float3> dacc(n);
        double k = time_device([&] { nbody_step<<<n / NBODY_TILE, NBODY_TILE>>>(dpos, dvel, dnp, dnv, dacc, n, dt); });
        auto gp = dnp.download();
        auto gv = dnv.download();
        auto ga = dacc.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            // Tolerance relative to the sum of the magnitudes of the terms.
            double tol = 2e-5 * acc_scale[i] + 1e-4;
            float g[3] = {ga[i].x, ga[i].y, ga[i].z};
            for (int c = 0; c < 3; ++c)
                if (std::fabs(g[c] - want_acc[3 * i + c]) > tol) {
                    std::printf("  mismatch acc[%d].%d: got %.9g want %.9g tol %.3g\n", i, c, g[c], want_acc[3 * i + c], tol);
                    ok = false;
                }
            double vt = tol * dt + 1e-6, pt = vt * dt + 1e-5 * 10;
            if (std::fabs(gv[i].x - want_vel[i].x) > vt || std::fabs(gv[i].y - want_vel[i].y) > vt ||
                std::fabs(gv[i].z - want_vel[i].z) > vt) {
                std::printf("  mismatch vel[%d]\n", i);
                ok = false;
            }
            if (std::fabs(gp[i].x - want_pos[i].x) > pt || std::fabs(gp[i].y - want_pos[i].y) > pt ||
                std::fabs(gp[i].z - want_pos[i].z) > pt || gp[i].w != want_pos[i].w) {
                std::printf("  mismatch pos[%d]\n", i);
                ok = false;
            }
        }
        report(n == 4096 ? "nbody_allpairs_4096" : "nbody_allpairs_16384", ok, k, ref);
    }

    // Black-Scholes.
    {
        const int n = 1 << 22;
        const float R = 0.02f, V = 0.30f;
        auto S = random_floats(n, 21, 5.0f, 30.0f);
        auto X = random_floats(n, 22, 1.0f, 100.0f);
        auto T = random_floats(n, 23, 0.25f, 10.0f);
        std::vector<float> wc(n), wp(n);
        double ref = best_ms([&] {
#pragma omp parallel for simd schedule(static)
            for (int i = 0; i < n; ++i) {
                float s = S[i], x = X[i], t = T[i];
                float sqrtT = std::sqrt(t);
                float d1 = (std::log(s / x) + (R + 0.5f * V * V) * t) / (V * sqrtT);
                float d2 = d1 - V * sqrtT;
                float c1 = cnd_host(d1), c2 = cnd_host(d2);
                float e = std::exp(-R * t);
                wc[i] = s * c1 - x * e * c2;
                wp[i] = x * e * (1.0f - c2) - s * (1.0f - c1);
            }
        });
        Buffer<float> dS(S), dX(X), dT(T), dc(n), dp(n);
        double k = time_device([&] { black_scholes<<<480, 128>>>(dc, dp, dS, dX, dT, R, V, n); });
        auto gc = dc.download();
        auto gp = dp.download();
        report("black_scholes_call", close(gc.data(), wc.data(), n, 1e-4, 2e-4, "call"), k, ref);
        report("black_scholes_put", close(gp.data(), wp.data(), n, 1e-4, 2e-4, "put"), k, ref);
    }

    // Monte Carlo pi.
    {
        constexpr int SAMPLES = 256, BLOCK = 256;
        const int blocks = 2048;
        const uint32_t nthreads = blocks * BLOCK;
        unsigned long long want = 0;
        double ref = best_ms([&] {
            unsigned long long acc = 0;
#pragma omp parallel for reduction(+ : acc) schedule(static)
            for (uint32_t t = 0; t < nthreads; ++t) acc += mc_thread_hits<SAMPLES>(t);
            want = acc;
        });
        Buffer<unsigned long long> dtotal(1);
        double k = time_device([&] {
            STRESS_CHECK(cudaMemset(dtotal, 0, sizeof(unsigned long long)));
            monte_carlo_pi<SAMPLES, BLOCK><<<blocks, BLOCK>>>(dtotal);
        });
        auto got = dtotal.download();
        double pi = 4.0 * double(got[0]) / (double(nthreads) * SAMPLES);
        char note[64];
        std::snprintf(note, sizeof note, "pi=%.6f", pi);
        bool ok = got[0] == want && std::fabs(pi - 3.14159265) < 1e-3;
        if (got[0] != want) std::printf("  hits got %llu want %llu\n", got[0], want);
        report("monte_carlo_pi", ok, k, ref, note);
    }

    // Mandelbrot.
    {
        const int w = 1024, h = 1024, max_iter = 2048;
        const float x0 = -2.0f, y0 = -1.25f, dx = 2.5f / w, dy = 2.5f / h;
        std::vector<int> want(w * h);
        double ref = best_ms([&] {
#pragma omp parallel for schedule(dynamic, 4)
            for (int py = 0; py < h; ++py)
                for (int px = 0; px < w; ++px) {
                    float cx = x0 + px * dx, cy = y0 + py * dy;
                    float x = 0, y = 0;
                    int it = 0;
                    while (x * x + y * y <= 4.0f && it < max_iter) {
                        float xt = x * x - y * y + cx;
                        y = 2.0f * x * y + cy;
                        x = xt;
                        ++it;
                    }
                    want[py * w + px] = it;
                }
        });
        Buffer<int> dit(w * h);
        dim3 block(16, 16), grid(w / 16, h / 16);
        double k = time_device([&] { mandelbrot<<<grid, block>>>(dit, w, h, x0, y0, dx, dy, max_iter); });
        auto got = dit.download();
        // FMA contraction can change the escape iteration of pixels next to
        // the boundary of the set: accept a small fraction of mismatches.
        size_t bad = 0, inside = 0;
        for (int i = 0; i < w * h; ++i) {
            bad += got[i] != want[i];
            inside += want[i] == max_iter;
        }
        double frac = double(bad) / (w * h);
        char note[96];
        std::snprintf(note, sizeof note, "mismatch=%zu(%.4f%%) inside=%zu", bad, 100 * frac, inside);
        report("mandelbrot_1024", frac < 1e-3 && inside > 0, k, ref, note);
    }

    return finish();
}
