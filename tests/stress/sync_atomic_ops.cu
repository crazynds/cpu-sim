// The other global atomics: atomicSub, atomicExch (int, float), atomicMin /
// atomicMax (int, unsigned, long long), atomicAnd / Or / Xor, atomicInc /
// atomicDec with their wrap-around, atomicCAS (int, unsigned long long) and
// CAS loops: a float max and a lock-free linked-list push.
#include "stress.cuh"

#include <map>

constexpr int BS = 256;
inline int blocks_for(int n) { return (n + BS - 1) / BS; }

__global__ void sub_kernel(const int* v, int* bins, int n, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicSub(&bins[i % nb], v[i]);
}

// Each thread swaps its own value into a slot; the old values it gets back
// and the final content form the chain of every value ever stored there.
__global__ void exch_int(int* slots, int* olds, int n, int nslots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) olds[i] = atomicExch(&slots[i % nslots], i);
}

__global__ void exch_float(float* slots, float* olds, int n, int nslots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) olds[i] = atomicExch(&slots[(i / 3) % nslots], float(i) * 0.5f);
}

__global__ void minmax_int(const int* v, int* mn, int* mx, int n, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicMin(&mn[i % nb], v[i]);
        atomicMax(&mx[i % nb], v[i]);
    }
}

__global__ void minmax_unsigned(const unsigned* v, unsigned* mn, unsigned* mx, int n, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicMin(&mn[v[i] % nb], v[i]);
        atomicMax(&mx[v[i] % nb], v[i]);
    }
}

__global__ void minmax_ll(const long long* v, long long* mn, long long* mx, long long* olds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        olds[i] = atomicMin(&mn[0], v[i]);
        atomicMax(&mx[0], v[i]);
    }
}

__global__ void bitwise(const unsigned* v, unsigned* a, unsigned* o, unsigned* x, int* xi, int n, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int b = i % nb;
        atomicAnd(&a[b], v[i] | 0x0f0f0f0fu);
        atomicOr(&o[b], v[i] & 0x80402010u);
        atomicXor(&x[b], v[i]);
        atomicXor(&xi[b], int(v[i] >> 3));
    }
}

__global__ void inc_kernel(unsigned* c, unsigned limit, unsigned* olds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) olds[i] = atomicInc(c, limit);
}

__global__ void dec_kernel(unsigned* c, unsigned limit, unsigned* olds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) olds[i] = atomicDec(c, limit);
}

__global__ void cas_counter(int* bins, const unsigned* v, int n, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int* p = &bins[v[i] % nb];
        int old = *p, assumed;
        do {
            assumed = old;
            old = atomicCAS(p, assumed, assumed + int(v[i] & 15));
        } while (old != assumed);
    }
}

// Arg max in one 64-bit word: value in the high half, ~index in the low half
// (so ties go to the smallest index), updated with a CAS loop.
__global__ void cas_argmax(const unsigned* v, unsigned long long* best, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long mine = ((unsigned long long)(v[i] >> 12) << 32) | (unsigned)~i;
        unsigned long long old = *best;
        while (old < mine) {
            unsigned long long prev = atomicCAS(best, old, mine);
            if (prev == old) break;
            old = prev;
        }
    }
}

__device__ float atomicMaxFloat(float* addr, float value) {
    int* p = reinterpret_cast<int*>(addr);
    int old = *p;
    while (__int_as_float(old) < value) {
        int assumed = old;
        old = atomicCAS(p, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}

__global__ void float_max_kernel(const float* v, float* out, int n, int nb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMaxFloat(&out[i % nb], v[i]);
}

// Lock-free push onto one of `nlists` singly linked lists (Treiber stack).
__global__ void push_kernel(int* heads, int* next, int n, int nlists) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int* head = &heads[i % nlists];
        int old = *(volatile int*)head;
        while (true) {
            next[i] = old;
            int prev = atomicCAS(head, old, i);
            if (prev == old) break;
            old = prev;
        }
    }
}

// Checks that `olds` plus the final values are the history of each slot.
template <class T, class Slot>
bool exch_chain_ok(const std::vector<T>& olds, const std::vector<T>& finals, T init, int n, Slot slot_of,
                   std::function<T(int)> value_of) {
    int ns = int(finals.size());
    std::vector<std::vector<T>> got(ns), want(ns);
    for (int s = 0; s < ns; ++s) want[s].push_back(init), got[s].push_back(finals[s]);
    for (int i = 0; i < n; ++i) {
        got[slot_of(i)].push_back(olds[i]);
        want[slot_of(i)].push_back(value_of(i));
    }
    for (int s = 0; s < ns; ++s) {
        std::sort(got[s].begin(), got[s].end());
        std::sort(want[s].begin(), want[s].end());
        if (got[s] != want[s]) {
            std::printf("  slot %d history differs\n", s);
            return false;
        }
    }
    return true;
}

int main() {
    using namespace stress;

    {
        const int n = 1 << 20, nb = 1000;
        auto r = random_u32(n, 21);
        std::vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = int(r[i] % 2001) - 1000;
        std::vector<int> init(nb);
        for (int b = 0; b < nb; ++b) init[b] = b * 1000;
        Buffer<int> dv(v), bins(nb);
        double k = time_device([&] {
            bins.upload(init);
            sub_kernel<<<blocks_for(n), BS>>>(dv, bins, n, nb);
        });
        std::vector<int> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int s = init[b];
                for (int i = b; i < n; i += nb) s -= v[i];
                want[b] = s;
            }
        });
        auto got = bins.download();
        report("sub_int_bins", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        const int n = 1 << 18, ns = 61;
        Buffer<int> slots(ns), olds(n);
        std::vector<int> init(ns, -1);
        double k = time_device([&] {
            slots.upload(init);
            exch_int<<<blocks_for(n), BS>>>(slots, olds, n, ns);
        });
        bool ok = exch_chain_ok<int>(olds.download(), slots.download(), -1, n, [&](int i) { return i % ns; },
                                     [](int i) { return i; });
        report("exch_int_chain", ok, k, -1);
    }

    {
        const int n = 1 << 18, ns = 37;
        Buffer<float> slots(ns), olds(n);
        std::vector<float> init(ns, -2.0f);
        double k = time_device([&] {
            slots.upload(init);
            exch_float<<<blocks_for(n), BS>>>(slots, olds, n, ns);
        });
        bool ok = exch_chain_ok<float>(olds.download(), slots.download(), -2.0f, n,
                                       [&](int i) { return (i / 3) % ns; }, [](int i) { return float(i) * 0.5f; });
        report("exch_float_chain", ok, k, -1);
    }

    {
        const int n = 1 << 20, nb = 257;
        auto r = random_u32(n, 22);
        std::vector<int> v(r.begin(), r.end());
        Buffer<int> dv(v), mn(nb), mx(nb);
        std::vector<int> imin(nb, INT32_MAX), imax(nb, INT32_MIN);
        double k = time_device([&] {
            mn.upload(imin);
            mx.upload(imax);
            minmax_int<<<blocks_for(n), BS>>>(dv, mn, mx, n, nb);
        });
        std::vector<int> wmn(nb), wmx(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                int a = INT32_MAX, z = INT32_MIN;
                for (int i = b; i < n; i += nb) a = std::min(a, v[i]), z = std::max(z, v[i]);
                wmn[b] = a, wmx[b] = z;
            }
        });
        auto gmn = mn.download(), gmx = mx.download();
        report("minmax_int", equal(gmn.data(), wmn.data(), nb, "min") && equal(gmx.data(), wmx.data(), nb, "max"), k,
               ref);
    }

    {
        const int n = 1 << 20, nb = 64;
        auto v = random_u32(n, 23);
        Buffer<unsigned> dv(v), mn(nb), mx(nb);
        std::vector<unsigned> imin(nb, 0xffffffffu), imax(nb, 0u);
        double k = time_device([&] {
            mn.upload(imin);
            mx.upload(imax);
            minmax_unsigned<<<blocks_for(n), BS>>>(dv, mn, mx, n, nb);
        });
        std::vector<unsigned> wmn(nb), wmx(nb);
        double ref = best_ms([&] {
            wmn = imin, wmx = imax;
#pragma omp parallel
            {
                std::vector<unsigned> a(imin), z(imax);
#pragma omp for nowait
                for (int i = 0; i < n; ++i) {
                    unsigned b = v[i] % nb;
                    a[b] = std::min(a[b], v[i]), z[b] = std::max(z[b], v[i]);
                }
#pragma omp critical
                for (int b = 0; b < nb; ++b) wmn[b] = std::min(wmn[b], a[b]), wmx[b] = std::max(wmx[b], z[b]);
            }
        });
        auto gmn = mn.download(), gmx = mx.download();
        report("minmax_unsigned", equal(gmn.data(), wmn.data(), nb, "min") && equal(gmx.data(), wmx.data(), nb, "max"),
               k, ref);
    }

    {  // long long, one address; the returned olds of atomicMin never go below the final min.
        const int n = 1 << 18;
        auto r = random_u32(n, 24), r2 = random_u32(n, 25);
        std::vector<long long> v(n);
        for (int i = 0; i < n; ++i) v[i] = (long long)((uint64_t(r[i]) << 32) | r2[i]);  // full range, negatives too
        Buffer<long long> dv(v), mn(1), mx(1), olds(n);
        std::vector<long long> imin{INT64_MAX}, imax{INT64_MIN};
        double k = time_device([&] {
            mn.upload(imin);
            mx.upload(imax);
            minmax_ll<<<blocks_for(n), BS>>>(dv, mn, mx, olds, n);
        });
        long long wmn = 0, wmx = 0;
        double ref = best_ms([&] {
            long long a = INT64_MAX, z = INT64_MIN;
#pragma omp parallel for reduction(min : a) reduction(max : z)
            for (int i = 0; i < n; ++i) a = std::min(a, v[i]), z = std::max(z, v[i]);
            wmn = a, wmx = z;
        });
        auto o = olds.download();
        bool ok = mn.download()[0] == wmn && mx.download()[0] == wmx;
        for (int i = 0; ok && i < n; ++i)
            if (o[i] < wmn) ok = false;
        report("minmax_longlong", ok, k, ref);
    }

    {
        const int n = 1 << 20, nb = 33;
        auto v = random_u32(n, 26);
        Buffer<unsigned> dv(v), a(nb), o(nb), x(nb);
        Buffer<int> xi(nb);
        std::vector<unsigned> ones(nb, 0xffffffffu), zeros(nb, 0u);
        std::vector<int> izeros(nb, 0x1234);
        double k = time_device([&] {
            a.upload(ones);
            o.upload(zeros);
            x.upload(zeros);
            xi.upload(izeros);
            bitwise<<<blocks_for(n), BS>>>(dv, a, o, x, xi, n, nb);
        });
        std::vector<unsigned> wa(nb), wo(nb), wx(nb);
        std::vector<int> wxi(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                unsigned A = ~0u, O = 0, X = 0;
                int XI = 0x1234;
                for (int i = b; i < n; i += nb) A &= v[i] | 0x0f0f0f0fu, O |= v[i] & 0x80402010u, X ^= v[i], XI ^= int(v[i] >> 3);
                wa[b] = A, wo[b] = O, wx[b] = X, wxi[b] = XI;
            }
        });
        auto ga = a.download(), go = o.download(), gx = x.download();
        auto gxi = xi.download();
        bool ok = equal(ga.data(), wa.data(), nb, "and") && equal(go.data(), wo.data(), nb, "or") &&
                  equal(gx.data(), wx.data(), nb, "xor") && equal(gxi.data(), wxi.data(), nb, "xor_int");
        report("and_or_xor", ok, k, ref);
    }

    // atomicInc: old >= limit ? 0 : old + 1. atomicDec: (old == 0 || old > limit) ? limit : old - 1.
    // The sequence of returned values does not depend on which thread does
    // which step, so the sorted olds and the final value are deterministic.
    // Both start above the limit to exercise the reset branch.
    for (int which = 0; which < 2; ++which) {
        const int n = 100003;
        const unsigned limit = 37, start = 1000;
        Buffer<unsigned> c(1), olds(n);
        double k = time_device([&] {
            c.upload({start});
            if (which == 0) inc_kernel<<<blocks_for(n), BS>>>(c, limit, olds, n);
            else dec_kernel<<<blocks_for(n), BS>>>(c, limit, olds, n);
        });
        std::vector<unsigned> want(n);
        unsigned cur = start;
        for (int i = 0; i < n; ++i) {
            want[i] = cur;
            cur = which == 0 ? (cur >= limit ? 0 : cur + 1) : ((cur == 0 || cur > limit) ? limit : cur - 1);
        }
        auto got = olds.download();
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        bool ok = c.download()[0] == cur && equal(got.data(), want.data(), n);
        report(which == 0 ? "inc_wraparound" : "dec_wraparound", ok, k, -1);
    }

    {
        const int n = 1 << 16, nb = 256;
        auto v = random_u32(n, 27);
        Buffer<unsigned> dv(v);
        Buffer<int> bins(nb);
        double k = time_device([&] {
            bins.zero();
            cas_counter<<<blocks_for(n), BS>>>(bins, dv, n, nb);
        });
        std::vector<int> want(nb);
        double ref = best_ms([&] {
            std::fill(want.begin(), want.end(), 0);
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
#pragma omp atomic
                want[v[i] % nb] += int(v[i] & 15);
            }
        });
        auto got = bins.download();
        report("cas_int_loop_bins", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        const int n = 1 << 20;
        auto v = random_u32(n, 28);
        Buffer<unsigned> dv(v);
        Buffer<unsigned long long> best(1);
        double k = time_device([&] {
            best.zero();
            cas_argmax<<<blocks_for(n), BS>>>(dv, best, n);
        });
        unsigned long long want = 0;
        double ref = best_ms([&] {
            unsigned long long b = 0;
#pragma omp parallel for reduction(max : b)
            for (int i = 0; i < n; ++i) b = std::max(b, ((unsigned long long)(v[i] >> 12) << 32) | (unsigned)~i);
            want = b;
        });
        report("cas_ull_argmax", best.download()[0] == want, k, ref);
    }

    {
        const int n = 1 << 20, nb = 128;
        auto v = random_floats(n, 29, -100.0f, 50.0f);
        Buffer<float> dv(v), out(nb);
        std::vector<float> init(nb, -INFINITY);
        double k = time_device([&] {
            out.upload(init);
            float_max_kernel<<<blocks_for(n), BS>>>(dv, out, n, nb);
        });
        std::vector<float> want(nb);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int b = 0; b < nb; ++b) {
                float m = -INFINITY;
                for (int i = b; i < n; i += nb) m = std::max(m, v[i]);
                want[b] = m;
            }
        });
        auto got = out.download();
        report("cas_float_max", equal(got.data(), want.data(), nb), k, ref);
    }

    {
        const int n = 1 << 14, nl = 32;
        Buffer<int> heads(nl), next(n);
        std::vector<int> empty(nl, -1);
        double k = time_device([&] {
            heads.upload(empty);
            push_kernel<<<blocks_for(n), BS>>>(heads, next, n, nl);
        });
        auto h = heads.download(), nx = next.download();
        std::vector<char> seen(n, 0);
        bool ok = true;
        for (int l = 0; l < nl && ok; ++l) {
            int len = 0;
            for (int p = h[l]; p != -1 && ok; p = nx[p], ++len) {
                if (p < 0 || p >= n || seen[p] || p % nl != l || len > n) {
                    std::printf("  list %d broken at node %d\n", l, p);
                    ok = false;
                } else seen[p] = 1;
            }
            if (ok && len != n / nl) {
                std::printf("  list %d has %d nodes\n", l, len);
                ok = false;
            }
        }
        report("cas_lockfree_push", ok, k, -1);
    }

    return finish();
}
