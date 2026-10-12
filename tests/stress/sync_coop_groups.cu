// Cooperative groups inside a block: this_thread_block, block.sync,
// thread_rank, tiled_partition<32> / <16> and the dynamic
// tiled_partition(block, 8), tile shuffles and votes, cg::reduce and
// cg::inclusive_scan / exclusive_scan, and coalesced_threads for
// warp-aggregated atomics.
#include "stress.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;

constexpr int BS = 256;

// Block tree reduction written with the group API only.
__global__ void block_reduce(const int* in, int* out) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ int s[BS];
    unsigned r = block.thread_rank();
    s[r] = in[blockIdx.x * block.size() + r];
    block.sync();
    for (unsigned k = block.size() / 2; k > 0; k >>= 1) {
        if (r < k) s[r] += s[r + k];
        block.sync();
    }
    if (r == 0) out[block.group_index().x] = s[0];
}

// Warp tiles: shfl_down reduction, one atomic per tile.
__global__ void tile32_shfl(const int* in, int* total, int* per_tile, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    int i = blockIdx.x * blockDim.x + block.thread_rank();
    int v = i < n ? in[i] : 0;
    for (int o = tile.size() / 2; o > 0; o /= 2) v += tile.shfl_down(v, o);
    if (tile.thread_rank() == 0) {
        atomicAdd(total, v);
        per_tile[blockIdx.x * (blockDim.x / 32) + tile.meta_group_rank()] = v;
    }
}

// Tiles of 16: cg::reduce with plus / greater / less and the tile votes.
__global__ void tile16_reduce(const float* in, float* sum, float* mx, int* mn, unsigned* ballot, int* anyall, int n) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<16>(block);
    int i = blockIdx.x * blockDim.x + block.thread_rank();
    float v = in[i];
    int q = int(v * 1000.0f);
    float s = cg::reduce(tile, v, cg::plus<float>());
    float m = cg::reduce(tile, v, cg::greater<float>());
    int lo = cg::reduce(tile, q, cg::less<int>());
    unsigned b = tile.ballot(v > 0.0f);
    int a = tile.any(v > 0.9f), l = tile.all(v > -0.9f);
    int t = i / 16;
    if (tile.thread_rank() == 0) sum[t] = s, mx[t] = m, mn[t] = lo, ballot[t] = b, anyall[t] = a * 2 + l;
}

// Dynamic tiles of 8 (thread_group): a shared-memory reduction per tile,
// synchronized with tile.sync().
__global__ void dynamic_tile8(const int* in, int* out) {
    __shared__ int s[BS];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_group tile = cg::tiled_partition(block, 8);
    unsigned r = tile.thread_rank();
    unsigned base = block.thread_rank() - r;
    s[block.thread_rank()] = in[blockIdx.x * BS + block.thread_rank()];
    tile.sync();
    for (unsigned k = tile.size() / 2; k > 0; k >>= 1) {
        if (r < k) s[base + r] += s[base + r + k];
        tile.sync();
    }
    out[blockIdx.x * BS + block.thread_rank()] = s[base] * 10 + int(r);
}

// Inclusive and exclusive scans over 32-thread tiles, then block-wide via
// a second tile scan over the per-tile totals.
__global__ void tile_scan(const int* in, int* incl, int* excl) {
    __shared__ int tot[BS / 32];
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int i = blockIdx.x * BS + block.thread_rank();
    int v = in[i];
    int x = cg::inclusive_scan(tile, v);
    int e = cg::exclusive_scan(tile, v, cg::plus<int>());
    if (tile.thread_rank() == 31) tot[tile.meta_group_rank()] = x;
    block.sync();
    if (tile.meta_group_rank() == 0) {
        int t = tile.thread_rank() < BS / 32 ? tot[tile.thread_rank()] : 0;
        t = cg::exclusive_scan(tile, t);
        if (tile.thread_rank() < BS / 32) tot[tile.thread_rank()] = t;
    }
    block.sync();
    incl[i] = x + tot[tile.meta_group_rank()];
    excl[i] = e + tot[tile.meta_group_rank()];
}

// Stream compaction with warp-aggregated atomics through coalesced_threads.
__global__ void coalesced_compact(const int* in, int* out, int* counter, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = in[i];
        if (v % 3 == 0) {
            cg::coalesced_group g = cg::coalesced_threads();
            int base;
            if (g.thread_rank() == 0) base = atomicAdd(counter, int(g.size()));
            base = g.shfl(base, 0);
            out[base + g.thread_rank()] = v;
        }
    }
}

// coalesced_threads in nested divergence: each active subset reduces itself.
__global__ void coalesced_reduce(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    int r = -1;
    if (v & 1) {
        auto g = cg::coalesced_threads();
        r = cg::reduce(g, v, cg::plus<int>()) * 64 + int(g.size());
    } else if (v & 2) {
        auto g = cg::coalesced_threads();
        r = g.shfl(v, g.size() - 1);  // the last active lane's value
    }
    out[i] = r;
}

std::vector<int> small_ints(int n, uint32_t seed, int mod) {
    auto r = stress::random_u32(n, seed);
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = int(r[i] % mod);
    return v;
}

int main() {
    using namespace stress;
    const int n = 1 << 20, nb = n / BS;

    {
        auto v = small_ints(n, 81, 100000);
        Buffer<int> dv(v), out(nb);
        double k = time_device([&] { block_reduce<<<nb, BS>>>(dv, out); });
        std::vector<int> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int s = 0;
                for (int t = 0; t < BS; ++t) s += v[b * BS + t];
                want[b] = s;
            }
        });
        auto got = out.download();
        report("cg_block_reduce", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        auto v = small_ints(n, 82, 2000);
        Buffer<int> dv(v), total(1), per(n / 32);
        double k = time_device([&] {
            total.zero();
            tile32_shfl<<<nb, BS>>>(dv, total, per, n);
        });
        std::vector<int> wper(n / 32);
        int want = 0;
        double ref = best_ms([&] {
            int s = 0;
#pragma omp parallel for reduction(+ : s)
            for (int w = 0; w < n / 32; ++w) {
                int t = 0;
                for (int l = 0; l < 32; ++l) t += v[w * 32 + l];
                wper[w] = t;
                s += t;
            }
            want = s;
        });
        auto got = per.download();
        report("cg_tile32_shfl_down", total.download()[0] == want && equal(got.data(), wper.data(), n / 32), k, ref);
    }

    {  // The float sum of 16 values depends on cg::reduce's internal order, so
       // it is checked against the double sum with 1e-6 of the magnitude sum;
       // max and min are exact.
        const int nt = n / 16;
        auto v = random_floats(n, 83);
        Buffer<float> dv(v), sum(nt), mx(nt);
        Buffer<int> mn(nt), anyall(nt);
        Buffer<unsigned> ballot(nt);
        double k = time_device([&] { tile16_reduce<<<nb, BS>>>(dv, sum, mx, mn, ballot, anyall, n); });
        std::vector<double> ws(nt), mag(nt);
        std::vector<float> wm(nt);
        std::vector<int> wmn(nt), waa(nt);
        std::vector<unsigned> wb(nt);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int t = 0; t < nt; ++t) {
                double s = 0, a = 0;
                float m = -INFINITY;
                int lo = INT_MAX, an = 0, al = 1;
                unsigned b = 0;
                for (int j = 0; j < 16; ++j) {
                    float x = v[t * 16 + j];
                    s += x, a += std::fabs(x), m = std::max(m, x), lo = std::min(lo, int(x * 1000.0f));
                    b |= unsigned(x > 0.0f) << j, an |= x > 0.9f, al &= x > -0.9f;
                }
                ws[t] = s, mag[t] = a, wm[t] = m, wmn[t] = lo, wb[t] = b, waa[t] = an * 2 + al;
            }
        });
        auto gs = sum.download(), gm = mx.download();
        auto gmn = mn.download(), gaa = anyall.download();
        auto gb = ballot.download();
        bool ok = equal(gm.data(), wm.data(), nt, "max") && equal(gmn.data(), wmn.data(), nt, "min") &&
                  equal(gb.data(), wb.data(), nt, "ballot") && equal(gaa.data(), waa.data(), nt, "any_all");
        for (int t = 0; ok && t < nt; ++t) {
            double g = gs[t];
            ok = close(&g, &ws[t], 1, 0, 1e-6 * mag[t], "sum");
        }
        report("cg_tile16_reduce_votes", ok, k, ref);
    }

    {
        auto v = small_ints(n, 84, 1000);
        Buffer<int> dv(v), out(n);
        double k = time_device([&] { dynamic_tile8<<<nb, BS>>>(dv, out); });
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int t = 0; t < n / 8; ++t) {
                int s = 0;
                for (int j = 0; j < 8; ++j) s += v[t * 8 + j];
                for (int j = 0; j < 8; ++j) want[t * 8 + j] = s * 10 + j;
            }
        });
        auto got = out.download();
        report("cg_dynamic_tile8", equal(got.data(), want.data(), n), k, ref);
    }

    {
        auto v = small_ints(n, 85, 1000);
        Buffer<int> dv(v), incl(n), excl(n);
        double k = time_device([&] { tile_scan<<<nb, BS>>>(dv, incl, excl); });
        std::vector<int> wi(n), we(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int s = 0;
                for (int t = 0; t < BS; ++t) {
                    we[b * BS + t] = s;
                    wi[b * BS + t] = s += v[b * BS + t];
                }
            }
        });
        auto gi = incl.download(), ge = excl.download();
        report("cg_tile_scans", equal(gi.data(), wi.data(), n, "incl") && equal(ge.data(), we.data(), n, "excl"), k, ref);
    }

    {
        auto v = small_ints(n, 86, 1 << 30);
        Buffer<int> dv(v), out(n), counter(1);
        double k = time_device([&] {
            counter.zero();
            coalesced_compact<<<nb, BS>>>(dv, out, counter, n);
        });
        std::vector<int> want;
        double ref = best_ms([&] {
            std::vector<std::vector<int>> parts(omp_get_max_threads());
#pragma omp parallel
            {
                auto& p = parts[omp_get_thread_num()];
                p.clear();
#pragma omp for
                for (int i = 0; i < n; ++i)
                    if (v[i] % 3 == 0) p.push_back(v[i]);
            }
            want.clear();
            for (auto& p : parts) want.insert(want.end(), p.begin(), p.end());
        });
        int c = counter.download()[0];
        auto o = out.download();
        std::vector<int> got(o.begin(), o.begin() + std::min(c, n));
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        report("cg_coalesced_compact", c == int(want.size()) && got == want, k, ref);
    }

    {
        auto v = small_ints(n, 87, 1000);
        Buffer<int> dv(v), out(n);
        double k = time_device([&] { coalesced_reduce<<<nb, BS>>>(dv, out, n); });
        std::vector<int> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int w = 0; w < n / 32; ++w) {
                int s = 0, c = 0, last = -1;
                for (int l = 0; l < 32; ++l) {
                    int x = v[w * 32 + l];
                    if (x & 1) s += x, ++c;
                    else if (x & 2) last = x;
                }
                for (int l = 0; l < 32; ++l) {
                    int x = v[w * 32 + l];
                    want[w * 32 + l] = (x & 1) ? s * 64 + c : (x & 2) ? last : -1;
                }
            }
        });
        auto got = out.download();
        report("cg_coalesced_divergent", equal(got.data(), want.data(), n), k, ref);
    }

    return finish();
}
