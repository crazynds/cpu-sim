#include "device/wmma_trace.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace cpusim {

namespace {

// Element types, as cpusim-devc encodes them (src/compiler/wmma_trace.cpp).
enum TypeCode { TOther = 0, TBf16 = 1, TS8 = 2, TU8 = 3, TF16 = 4, TF32 = 5, TS32 = 6 };

int esize(int type) {
    switch (type) {
        case TBf16: case TF16: return 2;
        case TS8: case TU8: return 1;
        case TF32: case TS32: return 4;
        default: return 0;
    }
}

// One stored 16 x 16 tile of C: where it went, where its sum started in A
// and B, how many 16-wide steps of k it took, and whether it added to C.
struct Tile {
    uintptr_t c = 0, a0 = 0, b0 = 0;
    uint32_t ldc = 0, lda = 0, ldb = 0, steps = 0;
    int type_ab = 0, type_c = 0;
    bool accumulate = false;
    bool operator==(const Tile& o) const {
        return c == o.c && a0 == o.a0 && b0 == o.b0 && ldc == o.ldc && lda == o.lda && ldb == o.ldb &&
               steps == o.steps && type_ab == o.type_ab && type_c == o.type_c && accumulate == o.accumulate;
    }
};

// The records of one warp (one lane of it).
struct Recorder {
    struct Frag {
        int kind = -1, type = 0, layout = 0;
        uintptr_t p = 0;
        uint32_t ldm = 0;
        // Accumulators:
        enum { Unset, Zero, FromC, Bad } init = Unset;
        uintptr_t c_src = 0;
        uint32_t c_ld = 0;
        uintptr_t a0 = 0, b0 = 0;
        uint32_t lda = 0, ldb = 0, steps = 0;
        int type_ab = 0;
        bool stored = false;
    };
    std::map<const void*, Frag> frags;
    std::vector<Tile> tiles;
    bool bad = false;
    void fail() { bad = true; }
};

}  // namespace

}  // namespace cpusim

using cpusim::Recorder;

// Called by the trace functions (exported: kernel modules resolve them in
// the process that loads them).
extern "C" {

__attribute__((visibility("default"), used)) void __cpusim_rec_load(void* r, const void* id, int kind, int type, int layout,
                                                                      const void* p, uint32_t ldm) {
    auto& rec = *static_cast<Recorder*>(r);
    Recorder::Frag& f = rec.frags[id];
    f.kind = kind, f.type = type, f.layout = layout, f.p = reinterpret_cast<uintptr_t>(p), f.ldm = ldm;
    if (kind == 2) {  // an accumulator loaded from C: C += A * B if it goes back to the same place
        f.init = layout == 0 ? Recorder::Frag::FromC : Recorder::Frag::Bad;
        f.c_src = f.p, f.c_ld = ldm, f.steps = 0, f.stored = false;
    }
}

__attribute__((visibility("default"), used)) void __cpusim_rec_fill(void* r, const void* id, double v) {
    auto& rec = *static_cast<Recorder*>(r);
    Recorder::Frag& f = rec.frags[id];
    f.kind = 2;
    f.init = v == 0.0 ? Recorder::Frag::Zero : Recorder::Frag::Bad;
    f.steps = 0, f.stored = false;
}

__attribute__((visibility("default"), used)) void __cpusim_rec_mma(void* r, const void* d, const void* a, const void* b, const void* c) {
    auto& rec = *static_cast<Recorder*>(r);
    if (d != c) return rec.fail();
    Recorder::Frag& acc = rec.frags[c];
    const Recorder::Frag& fa = rec.frags[a];
    const Recorder::Frag& fb = rec.frags[b];
    if (acc.kind != 2 || acc.init == Recorder::Frag::Unset || acc.init == Recorder::Frag::Bad || acc.stored)
        return rec.fail();
    // A and B row-major, of one type the AMX GEMM takes.
    if (fa.kind != 0 || fb.kind != 1 || fa.layout != 0 || fb.layout != 0 || fa.type != fb.type ||
        (fa.type != cpusim::TBf16 && fa.type != cpusim::TS8))
        return rec.fail();
    const int es = cpusim::esize(fa.type);
    if (acc.steps == 0) {
        acc.a0 = fa.p, acc.b0 = fb.p, acc.lda = fa.ldm, acc.ldb = fb.ldm, acc.type_ab = fa.type;
    } else if (fa.ldm != acc.lda || fb.ldm != acc.ldb || fa.type != acc.type_ab ||
               // Step s: 16 columns further in A's rows, 16 rows further down B.
               fa.p != acc.a0 + uintptr_t(acc.steps) * 16 * es ||
               fb.p != acc.b0 + uintptr_t(acc.steps) * 16 * acc.ldb * es) {
        return rec.fail();
    }
    ++acc.steps;
}

__attribute__((visibility("default"), used)) void __cpusim_rec_store(void* r, const void* id, int type, void* p, uint32_t ldm,
                                                                       int layout) {
    auto& rec = *static_cast<Recorder*>(r);
    Recorder::Frag& acc = rec.frags[id];
    const uintptr_t c = reinterpret_cast<uintptr_t>(p);
    if (acc.kind != 2 || acc.steps == 0 || acc.stored || layout != 0) return rec.fail();
    if (acc.init == Recorder::Frag::FromC && (acc.c_src != c || acc.c_ld != ldm)) return rec.fail();
    if (acc.init != Recorder::Frag::Zero && acc.init != Recorder::Frag::FromC) return rec.fail();
    // fp32 accumulation of bf16, int32 of int8.
    if ((acc.type_ab == cpusim::TBf16 && type != cpusim::TF32) || (acc.type_ab == cpusim::TS8 && type != cpusim::TS32))
        return rec.fail();
    cpusim::Tile t;
    t.c = c, t.ldc = ldm, t.a0 = acc.a0, t.b0 = acc.b0, t.lda = acc.lda, t.ldb = acc.ldb, t.steps = acc.steps;
    t.type_ab = acc.type_ab, t.type_c = type, t.accumulate = acc.init == Recorder::Frag::FromC;
    rec.tiles.push_back(t);
    acc.stored = true;
}

}  // extern "C"

namespace cpusim {

bool wmma_gemm_plan(Executor& ex, cpusim_trace_fn trace, const LaunchConfig& cfg, const uint8_t* params,
                    GemmArgs* out, const char** why) {
    auto no = [&](const char* w) {
        if (why) *why = w;
        return false;
    };
    const uint32_t bx = cfg.block[0], by = cfg.block[1], bz = cfg.block[2];
    const uint32_t threads = bx * by * bz, warps = (threads + 31) / 32;
    const uint64_t blocks = uint64_t(cfg.grid[0]) * cfg.grid[1] * cfg.grid[2];
    // Per worker: its tiles, and whether something did not fit.
    std::vector<std::vector<Tile>> found(ex.workers());
    std::vector<char> bad(ex.workers(), 0);
    auto run_lane = [&](uint64_t block, uint32_t t, Recorder& rec) {
        uint32_t s[14];
        s[0] = t % bx, s[1] = t / bx % by, s[2] = t / (bx * by);
        s[3] = uint32_t(block % cfg.grid[0]), s[4] = uint32_t(block / cfg.grid[0] % cfg.grid[1]),
        s[5] = uint32_t(block / (uint64_t(cfg.grid[0]) * cfg.grid[1]));
        s[6] = bx, s[7] = by, s[8] = bz;
        s[9] = cfg.grid[0], s[10] = cfg.grid[1], s[11] = cfg.grid[2];
        s[12] = t % 32, s[13] = 32;
        trace(params, s, &rec);
    };
    KernelFault fault = ex.run_native(
        blocks,
        [&](uint64_t block, int worker) {
            if (bad[size_t(worker)]) return;
            for (uint32_t w = 0; w < warps; ++w) {
                // The first and the last lane of the warp must agree (wmma's
                // arguments are the same for the whole warp).
                Recorder first, last;
                run_lane(block, w * 32, first);
                run_lane(block, std::min(threads - 1, w * 32 + 31), last);
                if (first.bad || last.bad || !(first.tiles == last.tiles)) {
                    bad[size_t(worker)] = 1;
                    return;
                }
                for (const auto& [id, f] : first.frags)
                    if (f.kind == 2 && f.steps && !f.stored) {  // computed but never stored
                        bad[size_t(worker)] = 1;
                        return;
                    }
                auto& v = found[size_t(worker)];
                v.insert(v.end(), first.tiles.begin(), first.tiles.end());
            }
        },
        false);
    if (fault.signal) return no("the trace faulted");
    for (char b : bad)
        if (b) return no("a warp does something other than one GEMM tile per accumulator");
    std::vector<Tile> tiles;
    for (auto& v : found) tiles.insert(tiles.end(), v.begin(), v.end());
    if (tiles.empty()) return no("no tile stored");

    // One matrix each for A, B and C: the same leading dimensions, types and
    // number of steps everywhere; bases where the smallest addresses are.
    const Tile& t0 = tiles.front();
    uintptr_t a = t0.a0, b = t0.b0, c = t0.c;
    for (const Tile& t : tiles) {
        if (t.lda != t0.lda || t.ldb != t0.ldb || t.ldc != t0.ldc || t.steps != t0.steps || t.type_ab != t0.type_ab ||
            t.type_c != t0.type_c || t.accumulate != t0.accumulate)
            return no("tiles of different shapes");
        a = std::min(a, t.a0), b = std::min(b, t.b0), c = std::min(c, t.c);
    }
    const uint64_t ea = esize(t0.type_ab), ec = esize(t0.type_c);
    const uint64_t K = uint64_t(t0.steps) * 16;
    uint64_t M = 0, N = 0;
    std::set<std::pair<uint64_t, uint64_t>> seen;
    for (const Tile& t : tiles) {
        const uint64_t oa = (t.a0 - a), ob = (t.b0 - b), oc = (t.c - c);
        if (oa % ea || ob % ea || oc % ec) return no("misaligned tiles");
        const uint64_t arow = oa / ea / t0.lda, acol = oa / ea % t0.lda;
        const uint64_t brow = ob / ea / t0.ldb, bcol = ob / ea % t0.ldb;
        const uint64_t crow = oc / ec / t0.ldc, ccol = oc / ec % t0.ldc;
        // The tile at (crow, ccol) is rows crow.. of A times columns ccol.. of
        // B, over k from 0: A's first column and B's first row.
        if (acol != 0 || brow != 0 || arow != crow || bcol != ccol || crow % 16 || ccol % 16)
            return no("a tile that is not its row of A times its column of B");
        if (!seen.insert({crow, ccol}).second) return no("a tile stored twice");
        M = std::max(M, crow + 16), N = std::max(N, ccol + 16);
    }
    if (seen.size() != (M / 16) * (N / 16)) return no("the tiles do not cover a rectangle");
    if (K > t0.lda || N > t0.ldb || N > t0.ldc) return no("leading dimensions smaller than the matrices");
    // C must not overlap A or B (the GEMM reads them after it writes C).
    auto end = [](uintptr_t base, uint64_t rows, uint64_t cols, uint64_t ld, uint64_t es) {
        return base + ((rows - 1) * ld + cols) * es;
    };
    const uintptr_t ce = end(c, M, N, t0.ldc, ec), ae = end(a, M, K, t0.lda, ea), be = end(b, K, N, t0.ldb, ea);
    if ((c < ae && a < ce) || (c < be && b < ce)) return no("C overlaps A or B");

    GemmArgs g;
    g.type = t0.type_ab == TS8 ? GemmType::S8_S32 : GemmType::BF16_F32;
    g.A = reinterpret_cast<const void*>(a), g.B = reinterpret_cast<const void*>(b), g.C = reinterpret_cast<void*>(c);
    g.lda = t0.lda, g.ldb = t0.ldb, g.ldc = t0.ldc;
    g.M = uint32_t(M), g.N = uint32_t(N), g.K = uint32_t(K);
    g.accumulate = t0.accumulate;
    if (!amx_gemm_supported(g)) return no("a shape the AMX GEMM does not take");
    *out = g;
    return true;
}

}  // namespace cpusim
