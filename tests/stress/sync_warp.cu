// Warp-level primitives: __shfl_sync / _up / _down / _xor with the width
// argument (including the documented out-of-segment behavior), shuffles of
// double and long long, __ballot_sync, __any_sync, __all_sync, __activemask,
// __match_any_sync, __match_all_sync, __reduce_*_sync (sm_80+), __syncwarp
// with full and partial masks, and shuffles in divergent code with partial
// masks.
#include "stress.cuh"

constexpr int BS = 256;
constexpr unsigned FULL = 0xffffffffu;

// ---- host models of the shuffles (lane-level, one warp) ----
// For a segment of `w` lanes, a lane may read from its own or an earlier
// segment; reading a later one (or past the warp) returns its own value.
static int seg_base(int lane, int w) { return lane & ~(w - 1); }
template <class T>
T h_idx(const T* x, int lane, int src, int w) { return x[seg_base(lane, w) + (src % w)]; }
template <class T>
T h_up(const T* x, int lane, int d, int w) { return (lane - seg_base(lane, w)) >= d ? x[lane - d] : x[lane]; }
template <class T>
T h_down(const T* x, int lane, int d, int w) { return (lane - seg_base(lane, w)) + d < w ? x[lane + d] : x[lane]; }
template <class T>
T h_xor(const T* x, int lane, int m, int w) {
    int j = lane ^ m;
    return j <= (seg_base(lane, w) | (w - 1)) ? x[j] : x[lane];
}

// Applies a per-warp host function over the whole array in parallel.
template <class F>
double for_warps(int n, F f) {
    return stress::best_ms([&] {
#pragma omp parallel for
        for (int w = 0; w < n / 32; ++w) f(w * 32);
    });
}

__global__ void shfl_idx_kernel(const int* v, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int x = v[i];
    out[0 * n + i] = __shfl_sync(FULL, x, (lane * 7 + 3) & 31);
    out[1 * n + i] = __shfl_sync(FULL, x, lane * 5 + 1, 8);    // srcLane beyond width: modulo width
    out[2 * n + i] = __shfl_sync(FULL, x, 15 - (lane & 15), 16);
    out[3 * n + i] = __shfl_sync(FULL, x, x & 31);             // data-dependent source lane
}

__global__ void shfl_up_down_kernel(const int* v, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int x = v[i];
    out[0 * n + i] = __shfl_up_sync(FULL, x, 3, 8);
    out[1 * n + i] = __shfl_down_sync(FULL, x, 3, 8);
    out[2 * n + i] = __shfl_down_sync(FULL, x, 5);
    out[3 * n + i] = __shfl_up_sync(FULL, x, 9, 16);
    int s = x;  // inclusive scan within segments of 16
    for (int d = 1; d < 16; d <<= 1) {
        int t = __shfl_up_sync(FULL, s, d, 16);
        if ((lane & 15) >= d) s += t;
    }
    out[4 * n + i] = s;
}

__global__ void shfl_xor_kernel(const int* v, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = v[i];
    int s = x;  // butterfly sum within segments of 8
    for (int m = 4; m > 0; m >>= 1) s += __shfl_xor_sync(FULL, s, m, 8);
    out[0 * n + i] = s;
    out[1 * n + i] = __shfl_xor_sync(FULL, x, 16, 16);  // later segment: own value; earlier: partner
    out[2 * n + i] = __shfl_xor_sync(FULL, x, 8, 8);
    out[3 * n + i] = __shfl_xor_sync(FULL, x, 13);
}

__global__ void shfl_wide_kernel(const double* d, const long long* q, double* dout, long long* qout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    double x = d[i];
    for (int m = 16; m > 0; m >>= 1) x += __shfl_xor_sync(FULL, x, m);
    long long y = q[i];
    for (int o = 16; o > 0; o >>= 1) y += __shfl_down_sync(FULL, y, o);
    dout[i] = x;
    dout[n + i] = __shfl_sync(FULL, d[i], 31 - lane);
    qout[i] = __shfl_sync(FULL, y, 0);  // lane 0 holds the warp total
    qout[n + i] = __shfl_up_sync(FULL, q[i], 1);
}

__global__ void vote_kernel(const float* v, unsigned* ballot, int* any, int* all, int* popc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = v[i];
    unsigned b = __ballot_sync(FULL, x > 0.0f);
    int a = __any_sync(FULL, x > 0.99f);
    int l = __all_sync(FULL, x > -0.99f);
    if ((threadIdx.x & 31) == 0) {
        int w = i / 32;
        ballot[w] = b, any[w] = a != 0, all[w] = l != 0, popc[w] = __popc(b);
    }
}

__global__ void activemask_kernel(const int* v, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned m = 0;
    if (v[i] % 3 == 0) {
        m = __activemask();
        int leader = __ffs(m) - 1;
        m = __shfl_sync(m, m, leader);  // use the mask as a real member mask
    }
    out[i] = m;
}

__global__ void match_kernel(const int* v, unsigned* any_out, unsigned* all_out, int* pred_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int key = v[i] % 5;
    any_out[i] = __match_any_sync(FULL, key);
    long long wide = (long long)(v[i] / 1000) << 33;  // warp-uniform for some warps
    int pred = -1;
    all_out[i] = __match_all_sync(FULL, wide, &pred);
    pred_out[i] = pred;
}

__global__ void reduce_sync_kernel(const unsigned* v, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    unsigned x = v[i];
    out[0 * n + i] = __reduce_add_sync(FULL, x & 0xffff);
    out[1 * n + i] = (unsigned)__reduce_min_sync(FULL, (int)x);
    out[2 * n + i] = (unsigned)__reduce_max_sync(FULL, (int)x);
    out[3 * n + i] = __reduce_min_sync(FULL, x);
    out[4 * n + i] = __reduce_max_sync(FULL, x);
    out[5 * n + i] = __reduce_or_sync(FULL, x & (1u << (x % 32)));
    out[6 * n + i] = __reduce_and_sync(FULL, x | 0xf0f0f0f0u);
    out[7 * n + i] = __reduce_xor_sync(FULL, x);
    unsigned partial = 0;
    if (lane < 16) partial = __reduce_add_sync(0x0000ffffu, x >> 8);
    else if (lane & 1) partial = __reduce_max_sync(0xaaaa0000u, x);
    out[8 * n + i] = partial;
}

__global__ void syncwarp_kernel(const int* v, int* out, int n) {
    __shared__ int s[BS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, base = threadIdx.x & ~31;
    s[threadIdx.x] = v[i];
    __syncwarp();
    int a = s[base + 31 - lane];
    __syncwarp();
    s[threadIdx.x] = a * 2;
    __syncwarp();
    int b = s[base + ((lane + 1) & 31)];
    out[i] = a + b;
    __syncwarp();
    // Partial masks: the low half exchanges among itself twice.
    int c = 0;
    if (lane < 16) {
        s[threadIdx.x] = v[i] + 1;
        __syncwarp(0x0000ffffu);
        c = s[base + 15 - lane];
        __syncwarp(0x0000ffffu);
        s[threadIdx.x] = c * 3;
        __syncwarp(0x0000ffffu);
        c += s[base + ((lane + 3) & 15)];
    }
    out[n + i] = c;
}

// Odd and even lanes take different branches and shuffle only among
// themselves with partial masks.
__global__ void divergent_shfl_kernel(const int* v, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int x = v[i];
    int r;
    if (lane & 1) {
        const unsigned odd = 0xaaaaaaaau;
        r = x;
        for (int m = 2; m < 32; m <<= 1) r += __shfl_xor_sync(odd, r, m);
    } else {
        const unsigned even = 0x55555555u;
        r = __shfl_sync(even, x, 0) * 1000 + __shfl_down_sync(even, x, 2);  // lanes 30: own value
        if (lane < 8) r -= __shfl_sync(0x00000055u, x, 6);
    }
    out[i] = r;
}

// Inclusive block scan: shfl_up within warps, warp totals through shared.
__global__ void block_scan_kernel(const int* v, int* out, int n) {
    __shared__ int warp_tot[BS / 32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v[i];
    for (int d = 1; d < 32; d <<= 1) {
        int t = __shfl_up_sync(FULL, x, d);
        if (lane >= d) x += t;
    }
    if (lane == 31) warp_tot[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = lane < BS / 32 ? warp_tot[lane] : 0;
        for (int d = 1; d < BS / 32; d <<= 1) {
            int u = __shfl_up_sync(FULL, t, d);
            if (lane >= d) t += u;
        }
        if (lane < BS / 32) warp_tot[lane] = t;
    }
    __syncthreads();
    if (w > 0) x += warp_tot[w - 1];
    out[i] = x;
}

std::vector<int> small_ints(int n, uint32_t seed, int mod) {
    auto r = stress::random_u32(n, seed);
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = int(r[i] % mod);
    return v;
}

int main() {
    using namespace stress;
    const int n = 1 << 20, nb = n / BS, nw = n / 32;

    {
        auto v = small_ints(n, 61, 1 << 20);
        Buffer<int> dv(v), out(4 * n);
        double k = time_device([&] { shfl_idx_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(4 * n);
        double ref = for_warps(n, [&](int b) {
            const int* x = &v[b];
            for (int l = 0; l < 32; ++l) {
                want[0 * n + b + l] = h_idx(x, l, (l * 7 + 3) & 31, 32);
                want[1 * n + b + l] = h_idx(x, l, l * 5 + 1, 8);
                want[2 * n + b + l] = h_idx(x, l, 15 - (l & 15), 16);
                want[3 * n + b + l] = h_idx(x, l, x[l] & 31, 32);
            }
        });
        auto got = out.download();
        report("shfl_idx_widths", equal(got.data(), want.data(), 4 * n), k, ref);
    }

    {
        auto v = small_ints(n, 62, 1000);
        Buffer<int> dv(v), out(5 * n);
        double k = time_device([&] { shfl_up_down_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(5 * n);
        double ref = for_warps(n, [&](int b) {
            const int* x = &v[b];
            for (int l = 0; l < 32; ++l) {
                want[0 * n + b + l] = h_up(x, l, 3, 8);
                want[1 * n + b + l] = h_down(x, l, 3, 8);
                want[2 * n + b + l] = h_down(x, l, 5, 32);
                want[3 * n + b + l] = h_up(x, l, 9, 16);
                int s = 0;
                for (int j = l & ~15; j <= l; ++j) s += x[j];
                want[4 * n + b + l] = s;
            }
        });
        auto got = out.download();
        report("shfl_up_down_widths", equal(got.data(), want.data(), 5 * n), k, ref);
    }

    {
        auto v = small_ints(n, 63, 1000);
        Buffer<int> dv(v), out(4 * n);
        double k = time_device([&] { shfl_xor_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(4 * n);
        double ref = for_warps(n, [&](int b) {
            const int* x = &v[b];
            for (int l = 0; l < 32; ++l) {
                int s = 0;
                for (int j = l & ~7; j < (l & ~7) + 8; ++j) s += x[j];
                want[0 * n + b + l] = s;
                want[1 * n + b + l] = h_xor(x, l, 16, 16);
                want[2 * n + b + l] = h_xor(x, l, 8, 8);
                want[3 * n + b + l] = h_xor(x, l, 13, 32);
            }
        });
        auto got = out.download();
        report("shfl_xor_widths", equal(got.data(), want.data(), 4 * n), k, ref);
    }

    {  // The butterfly is replayed on the host in the same order: bit-exact.
        auto f = random_floats(n, 64);
        auto r = random_u32(n, 65);
        std::vector<double> d(n);
        std::vector<long long> q(n);
        for (int i = 0; i < n; ++i) d[i] = double(f[i]) / 3.0, q[i] = (long long)r[i] * ((i & 1) ? -40503 : 40503);
        Buffer<double> dd(d), dout(2 * n);
        Buffer<long long> dq(q), qout(2 * n);
        double k = time_device([&] { shfl_wide_kernel<<<nb, BS>>>(dd, dq, dout, qout, n); });
        std::vector<double> wd(2 * n);
        std::vector<long long> wq(2 * n);
        double ref = for_warps(n, [&](int b) {
            double x[32], y[32];
            for (int l = 0; l < 32; ++l) x[l] = d[b + l];
            for (int m = 16; m > 0; m >>= 1) {
                for (int l = 0; l < 32; ++l) y[l] = x[l] + x[l ^ m];
                std::memcpy(x, y, sizeof x);
            }
            long long t = 0;
            for (int l = 0; l < 32; ++l) t += q[b + l];
            for (int l = 0; l < 32; ++l) {
                wd[b + l] = x[l];
                wd[n + b + l] = d[b + 31 - l];
                wq[b + l] = t;
                wq[n + b + l] = h_up(&q[b], l, 1, 32);
            }
        });
        auto gd = dout.download();
        auto gq = qout.download();
        report("shfl_double_longlong", equal(gd.data(), wd.data(), 2 * n, "double") && equal(gq.data(), wq.data(), 2 * n, "ll"),
               k, ref);
    }

    {
        auto v = random_floats(n, 66);
        for (int i = 0; i < n; i += 32 * 5) v[i + 7] = 0.995f;  // some warps with any() true
        for (int w = 0; w < nw; w += 3)
            for (int l = 0; l < 32; ++l) v[w * 32 + l] = std::fabs(v[w * 32 + l]);  // some all-positive warps
        Buffer<float> dv(v);
        Buffer<unsigned> ballot(nw);
        Buffer<int> any(nw), all(nw), popc(nw);
        double k = time_device([&] { vote_kernel<<<nb, BS>>>(dv, ballot, any, all, popc, n); });
        std::vector<unsigned> wb(nw);
        std::vector<int> wa(nw), wl(nw), wp(nw);
        double ref = for_warps(n, [&](int b) {
            unsigned m = 0;
            int a = 0, l = 1;
            for (int j = 0; j < 32; ++j) m |= unsigned(v[b + j] > 0.0f) << j, a |= v[b + j] > 0.99f, l &= v[b + j] > -0.99f;
            wb[b / 32] = m, wa[b / 32] = a, wl[b / 32] = l, wp[b / 32] = __builtin_popcount(m);
        });
        auto gb = ballot.download();
        auto ga = any.download(), gl = all.download(), gp = popc.download();
        bool ok = equal(gb.data(), wb.data(), nw, "ballot") && equal(ga.data(), wa.data(), nw, "any") &&
                  equal(gl.data(), wl.data(), nw, "all") && equal(gp.data(), wp.data(), nw, "popc");
        report("ballot_any_all", ok, k, ref);
    }

    {
        auto v = small_ints(n, 67, 1000);
        Buffer<int> dv(v);
        Buffer<unsigned> out(n);
        double k = time_device([&] { activemask_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<unsigned> want(n);
        double ref = for_warps(n, [&](int b) {
            unsigned m = 0;
            for (int j = 0; j < 32; ++j) m |= unsigned(v[b + j] % 3 == 0) << j;
            for (int j = 0; j < 32; ++j) want[b + j] = v[b + j] % 3 == 0 ? m : 0;
        });
        auto got = out.download();
        report("activemask_divergent", equal(got.data(), want.data(), n), k, ref);
    }

    {
        auto v = small_ints(n, 68, 1 << 20);
        for (int w = 0; w < nw; w += 4)
            for (int l = 0; l < 32; ++l) v[w * 32 + l] = 5000 + (v[w * 32 + l] % 1000);  // v/1000 uniform
        Buffer<int> dv(v), pred(n);
        Buffer<unsigned> anyo(n), allo(n);
        double k = time_device([&] { match_kernel<<<nb, BS>>>(dv, anyo, allo, pred, n); });
        std::vector<unsigned> wany(n), wall(n);
        std::vector<int> wpred(n);
        double ref = for_warps(n, [&](int b) {
            bool same = true;
            for (int j = 0; j < 32; ++j) same &= v[b + j] / 1000 == v[b] / 1000;
            for (int l = 0; l < 32; ++l) {
                unsigned m = 0;
                for (int j = 0; j < 32; ++j) m |= unsigned(v[b + j] % 5 == v[b + l] % 5) << j;
                wany[b + l] = m;
                wall[b + l] = same ? FULL : 0u;
                wpred[b + l] = same;
            }
        });
        auto ga = anyo.download(), gl = allo.download();
        auto gp = pred.download();
        bool ok = equal(ga.data(), wany.data(), n, "match_any") && equal(gl.data(), wall.data(), n, "match_all") &&
                  equal(gp.data(), wpred.data(), n, "pred");
        report("match_any_all", ok, k, ref);
    }

    {
        auto v = random_u32(n, 69);
        Buffer<unsigned> dv(v), out(9 * n);
        double k = time_device([&] { reduce_sync_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<unsigned> want(9 * n);
        double ref = for_warps(n, [&](int b) {
            unsigned add = 0, umn = ~0u, umx = 0, o = 0, a = ~0u, x = 0, padd = 0, pmax = 0;
            int mn = INT_MAX, mx = INT_MIN;
            for (int j = 0; j < 32; ++j) {
                unsigned u = v[b + j];
                add += u & 0xffff, umn = std::min(umn, u), umx = std::max(umx, u);
                mn = std::min(mn, int(u)), mx = std::max(mx, int(u));
                o |= u & (1u << (u % 32)), a &= u | 0xf0f0f0f0u, x ^= u;
                if (j < 16) padd += u >> 8;
                else if (j & 1) pmax = std::max(pmax, u);
            }
            for (int l = 0; l < 32; ++l) {
                unsigned* w = &want[b + l];
                w[0] = add, w[n] = unsigned(mn), w[2 * n] = unsigned(mx), w[3 * n] = umn, w[4 * n] = umx;
                w[5 * n] = o, w[6 * n] = a, w[7 * n] = x;
                w[8 * n] = l < 16 ? padd : (l & 1) ? pmax : 0;
            }
        });
        auto got = out.download();
        report("reduce_sync_ops", equal(got.data(), want.data(), 9 * n), k, ref);
    }

    {
        auto v = small_ints(n, 70, 10000);
        Buffer<int> dv(v), out(2 * n);
        double k = time_device([&] { syncwarp_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(2 * n);
        double ref = for_warps(n, [&](int b) {
            const int* x = &v[b];
            for (int l = 0; l < 32; ++l) {
                int a = x[31 - l], bb = 2 * x[31 - ((l + 1) & 31)];
                want[b + l] = a + bb;
                int c = 0;
                if (l < 16) {
                    int j = (l + 3) & 15;
                    c = (x[15 - l] + 1) + 3 * (x[15 - j] + 1);
                }
                want[n + b + l] = c;
            }
        });
        auto got = out.download();
        report("syncwarp_masks", equal(got.data(), want.data(), 2 * n), k, ref);
    }

    {
        auto v = small_ints(n, 71, 1000);
        Buffer<int> dv(v), out(n);
        double k = time_device([&] { divergent_shfl_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(n);
        double ref = for_warps(n, [&](int b) {
            const int* x = &v[b];
            int odd = 0;
            for (int j = 1; j < 32; j += 2) odd += x[j];
            for (int l = 0; l < 32; ++l) {
                if (l & 1) want[b + l] = odd;
                else {
                    int r = x[0] * 1000 + (l + 2 < 32 ? x[l + 2] : x[l]);
                    if (l < 8) r -= x[6];
                    want[b + l] = r;
                }
            }
        });
        auto got = out.download();
        report("shfl_divergent_partial_masks", equal(got.data(), want.data(), n), k, ref);
    }

    {
        auto v = small_ints(n, 72, 2001);
        for (auto& x : v) x -= 1000;
        Buffer<int> dv(v), out(n);
        double k = time_device([&] { block_scan_kernel<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int s = 0;
                for (int t = 0; t < BS; ++t) want[b * BS + t] = s += v[b * BS + t];
            }
        });
        auto got = out.download();
        report("shfl_up_block_scan", equal(got.data(), want.data(), n), k, ref);
    }

    return finish();
}
