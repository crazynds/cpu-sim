#include "device/amx_gemm.h"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cpuid.h>
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// How it works (docs/optimization/iteration-3/35):
//   1. B is packed into the tile layout AMX's second operand wants (VNNI:
//      pairs of bf16 or quadruples of int8 of consecutive k interleaved per
//      column), in panels of 32 columns: [n / 32][k / KT][2 tiles][16 rows][64 B].
//   2. A needs no new layout: its tiles are loaded from the caller's matrix
//      when every tile row starts a cache line (A 64-byte aligned, lda a
//      multiple of 64 bytes). That is twice as fast as packing it, which
//      cost more than it saved (4096^3 on a Xeon 6548Y+: 15 against 30
//      TFLOPS, 24 against 52 TOPS). Otherwise A is packed into pairs of
//      tiles: [m / 32][k / KT][32 rows][64 B]. Packed buffers are 2 MiB-
//      aligned: a tile row that straddles two cache lines halves the
//      throughput (25 against 43 TFLOPS).
//   3. Each unit of work is a 64 x 64 block of C, as four 32 x 32 blocks of
//      2 x 2 tiles accumulated over the whole of K (C never leaves the tiles
//      until its last tdp); units go down columns of blocks, so neighbours
//      share their B panel. Loads are skewed one tdp ahead (A0 B0 B1 tdp
//      A1 tdp tdp tdp), and each C tile is stored right after its last tdp.
//      Software prefetching and non-temporal tile loads made it slower on
//      Sapphire/Emerald Rapids and are not used.

namespace cpusim {

namespace {

constexpr size_t kAlign = size_t(2) << 20;

// Packing buffers, kept between calls (one GEMM after another reuses them).
struct Scratch {
    std::mutex mutex;
    uint8_t* a = nullptr;
    uint8_t* b = nullptr;
    size_t a_size = 0, b_size = 0;
};
Scratch& scratch() {
    static Scratch s;
    return s;
}
uint8_t* grow(uint8_t*& p, size_t& have, size_t need) {
    if (have >= need) return p;
    std::free(p);
    have = (need + kAlign - 1) / kAlign * kAlign;
    p = static_cast<uint8_t*>(std::aligned_alloc(kAlign, have));
    if (p) madvise(p, have, MADV_HUGEPAGE);
    else have = 0;
    return p;
}

struct alignas(64) TileConfig {
    uint8_t palette = 1, start = 0, reserved[14] = {};
    uint16_t colsb[16] = {};
    uint8_t rows[16] = {};
};

__attribute__((target("amx-tile"))) void configure_tiles() {
    TileConfig cfg;
    for (int i = 0; i < 8; ++i) cfg.colsb[i] = 64, cfg.rows[i] = 16;
    _tile_loadconfig(&cfg);
}

template <bool INT8>
struct Kernel {
    using Acc = std::conditional_t<INT8, int32_t, float>;
    static constexpr int KT = INT8 ? 64 : 32;  // k per tile step (64 bytes of a row)
    const GemmArgs& g;
    uint8_t *ap, *bp;
    uint32_t kts;
    // A's tiles: packed (2 KiB per tile step, A1 1 KiB after A0, rows 64
    // bytes apart) or read in place from the caller's matrix.
    const uint8_t* a_base = nullptr;
    size_t a_step = 2048, a_half = 1024, a_ld = 64, a_row32 = 0;

    uint8_t* apair(uint32_t m32, uint32_t kt) const { return ap + (size_t(m32) * kts + kt) * 2048; }
    uint8_t* bpanel(uint32_t n32, uint32_t kt) const { return bp + (size_t(n32) * kts + kt) * 2048; }

    // Unit: one (n32, kt) pair of B tiles, rows interleaved.
    __attribute__((target("avx512f,avx512bw,avx512dq,avx512vbmi"))) void pack_b(uint64_t u) const {
        const uint32_t n32 = uint32_t(u / kts), kt = uint32_t(u % kts);
        uint8_t* dst = bpanel(n32, kt);
        if (!INT8) {
            alignas(64) static const uint16_t idx_init[32] = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                                                              8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
            const __m512i idx = _mm512_load_si512(idx_init);
            const uint16_t* B = static_cast<const uint16_t*>(g.B);
            for (int h = 0; h < 2; ++h)
                for (int q = 0; q < 16; ++q) {
                    const size_t k = size_t(kt) * KT + 2 * q, n = size_t(n32) * 32 + h * 16;
                    const __m512i r0 = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + k * g.ldb + n)));
                    const __m512i r1 = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + (k + 1) * g.ldb + n)));
                    _mm512_store_si512(dst + (h * 16 + q) * 64, _mm512_permutex2var_epi16(r0, idx, r1));
                }
        } else {
            alignas(64) static const uint8_t idx_init[64] = {
                0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53,
                6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63};
            const __m512i idx = _mm512_load_si512(idx_init);
            const int8_t* B = static_cast<const int8_t*>(g.B);
            for (int h = 0; h < 2; ++h)
                for (int q = 0; q < 16; ++q) {
                    const size_t k = size_t(kt) * KT + 4 * q, n = size_t(n32) * 32 + h * 16;
                    __m512i rows = _mm512_castsi128_si512(_mm_loadu_si128(reinterpret_cast<const __m128i*>(B + k * g.ldb + n)));
                    rows = _mm512_inserti64x2(rows, _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + (k + 1) * g.ldb + n)), 1);
                    rows = _mm512_inserti64x2(rows, _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + (k + 2) * g.ldb + n)), 2);
                    rows = _mm512_inserti64x2(rows, _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + (k + 3) * g.ldb + n)), 3);
                    _mm512_store_si512(dst + (h * 16 + q) * 64, _mm512_permutexvar_epi8(idx, rows));
                }
        }
    }

    // Unit: 32 rows of A (m32) over a block of 8 tile steps: each row gives
    // 512 contiguous bytes to 8 places 2 KiB apart. (A row at a time wrote
    // 128 lines 2 KiB apart, all in two L1 sets; 64 bytes of each of 32 rows
    // read K elements apart, the same on the reads. One unit per m32 left
    // too few units to spread over the workers.)
    static constexpr uint32_t kAKts = 8;
    uint32_t a_blocks() const { return (kts + kAKts - 1) / kAKts; }
    __attribute__((target("avx512f"))) void pack_a(uint64_t u) const {
        const uint32_t m32 = uint32_t(u / a_blocks()), kt0 = uint32_t(u % a_blocks()) * kAKts;
        const uint32_t kt1 = std::min(kts, kt0 + kAKts);
        const size_t esize = INT8 ? 1 : 2;
        const uint8_t* A = static_cast<const uint8_t*>(g.A) + size_t(m32) * 32 * g.lda * esize;
        uint8_t* base = apair(m32, 0);
        for (int r = 0; r < 32; ++r) {
            const uint8_t* src = A + r * g.lda * esize + size_t(kt0) * 64;
            uint8_t* dst = base + size_t(kt0) * 2048 + r * 64;
            for (uint32_t kt = kt0; kt < kt1; ++kt, src += 64, dst += 2048)
                _mm512_store_si512(dst, _mm512_loadu_si512(src));
        }
    }

    __attribute__((target("amx-tile,amx-bf16,amx-int8"))) static void tdp(int which) {
        // Tile numbers are immediates: C(i, j) = tile 2i + j, A(i) = 4 + i, B(j) = 6 + j.
        if (INT8) {
            switch (which) {
                case 0: _tile_dpbssd(0, 4, 6); break;
                case 1: _tile_dpbssd(1, 4, 7); break;
                case 2: _tile_dpbssd(2, 5, 6); break;
                default: _tile_dpbssd(3, 5, 7); break;
            }
        } else {
            switch (which) {
                case 0: _tile_dpbf16ps(0, 4, 6); break;
                case 1: _tile_dpbf16ps(1, 4, 7); break;
                case 2: _tile_dpbf16ps(2, 5, 6); break;
                default: _tile_dpbf16ps(3, 5, 7); break;
            }
        }
    }

    // One 32 x 32 block of C at (m0, n0) over the whole of K.
    __attribute__((target("amx-tile,amx-bf16,amx-int8"))) void block(uint32_t m0, uint32_t n0) const {
        Acc* c = static_cast<Acc*>(g.C) + size_t(m0) * g.ldc + n0;
        const size_t ldc = g.ldc * sizeof(Acc);
        const uint8_t* a = a_base + size_t(m0 / 32) * a_row32;
        const uint8_t* b = bpanel(n0 / 32, 0);
        if (g.accumulate) {
            _tile_loadd(0, c, ldc);
            _tile_loadd(1, c + 16, ldc);
            _tile_loadd(2, c + 16 * g.ldc, ldc);
            _tile_loadd(3, c + 16 * g.ldc + 16, ldc);
        } else {
            _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
        }
        const size_t step = a_step, half = a_half, ld = a_ld;
        for (uint32_t i = 0; i + 1 < kts; ++i, a += step, b += 2048) {
            _tile_loadd(4, a, ld); _tile_loadd(6, b, 64); _tile_loadd(7, b + 1024, 64);
            tdp(0);
            _tile_loadd(5, a + half, ld);
            tdp(1); tdp(2); tdp(3);
        }
        _tile_loadd(4, a, ld); _tile_loadd(6, b, 64); _tile_loadd(7, b + 1024, 64);
        tdp(0); _tile_stored(0, c, ldc);
        _tile_loadd(5, a + half, ld);
        tdp(1); _tile_stored(1, c + 16, ldc);
        tdp(2); _tile_stored(2, c + 16 * g.ldc, ldc);
        tdp(3); _tile_stored(3, c + 16 * g.ldc + 16, ldc);
    }

    // Unit: a 64 x 64 block (or what is left of one at the edges, in 32s),
    // units down a column of blocks first.
    void compute(uint64_t u) const {
        const uint32_t mblocks = (g.M + 63) / 64;
        const uint32_t m64 = uint32_t(u % mblocks) * 64, n64 = uint32_t(u / mblocks) * 64;
        configure_tiles();
        for (uint32_t s = 0; s < 4; ++s) {
            const uint32_t m0 = m64 + (s & 1) * 32, n0 = n64 + (s >> 1) * 32;
            if (m0 < g.M && n0 < g.N) block(m0, n0);
        }
    }
};

template <bool INT8>
KernelFault run(Executor& ex, const GemmArgs& g) {
    Scratch& s = scratch();
    std::lock_guard lock(s.mutex);
    Kernel<INT8> k{g, nullptr, nullptr, g.K / Kernel<INT8>::KT};
    k.ap = grow(s.a, s.a_size, size_t(g.M) * g.K * (INT8 ? 1 : 2));  // (unused when A is read in place)
    k.bp = grow(s.b, s.b_size, size_t(g.K) * g.N * (INT8 ? 1 : 2));
    if (!k.ap || !k.bp) {
        KernelFault f;
        f.signal = SIGSEGV;
        return f;
    }
    static const bool verbose = std::getenv("CPUSIM_VERBOSE") != nullptr;
    const size_t esize = INT8 ? 1 : 2;
    const bool direct = reinterpret_cast<uintptr_t>(g.A) % 64 == 0 && g.lda * esize % 64 == 0;
    if (direct) {
        k.a_base = static_cast<const uint8_t*>(g.A);
        k.a_step = 64, k.a_ld = g.lda * esize, k.a_half = 16 * k.a_ld, k.a_row32 = 32 * k.a_ld;
    } else {
        k.a_base = k.ap, k.a_row32 = size_t(k.kts) * 2048;
    }
    auto t0 = std::chrono::steady_clock::now();
    KernelFault f = ex.run_native(uint64_t(g.N / 32) * k.kts, [&](uint64_t u, int) { k.pack_b(u); }, false);
    auto t1 = std::chrono::steady_clock::now();
    if (!f.signal && !direct) f = ex.run_native(uint64_t(g.M / 32) * k.a_blocks(), [&](uint64_t u, int) { k.pack_a(u); }, false);
    auto t2 = std::chrono::steady_clock::now();
    if (!f.signal)
        f = ex.run_native(uint64_t((g.M + 63) / 64) * ((g.N + 63) / 64), [&](uint64_t u, int) { k.compute(u); }, true);
    auto t3 = std::chrono::steady_clock::now();
    if (verbose) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::fprintf(stderr, "cpusim: AMX GEMM %ux%ux%u: pack B %.2f ms, pack A %.2f ms, compute %.2f ms\n", g.M, g.N,
                     g.K, ms(t0, t1), ms(t1, t2), ms(t2, t3));
    }
    return f;
}

bool os_amx_state() {
    // The executor asks for it when it starts; asking again only reports.
    constexpr long kReqXcompPerm = 0x1023, kXfeatureXtiledata = 18;
    static const bool ok = syscall(SYS_arch_prctl, kReqXcompPerm, kXfeatureXtiledata) == 0;
    return ok;
}

}  // namespace

bool amx_gemm_available(GemmType type) {
    if (std::getenv("CPUSIM_NO_AMX_GEMM")) return false;
    unsigned a, b, c, d;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, a, b, c, d);
    const bool tile = d >> 24 & 1, bf16 = d >> 22 & 1, int8 = d >> 25 & 1;
    if (!tile || (type == GemmType::BF16_F32 ? !bf16 : !int8)) return false;
    return os_amx_state();
}

bool amx_gemm_supported(const GemmArgs& g) {
    const uint32_t kt = g.type == GemmType::S8_S32 ? 64 : 32;
    return g.A && g.B && g.C && g.M && g.N && g.K && g.M % 32 == 0 && g.N % 32 == 0 && g.K % kt == 0 &&
           g.lda >= g.K && g.ldb >= g.N && g.ldc >= g.N;
}

KernelFault amx_gemm(Executor& ex, const GemmArgs& g) {
    return g.type == GemmType::S8_S32 ? run<true>(ex, g) : run<false>(ex, g);
}

}  // namespace cpusim
