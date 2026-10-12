// Byte workloads: counting a multi-byte pattern (overlapping matches, tiles
// staged in shared memory with a halo, warp ballot + block reduce +
// atomicAdd), run-length decoding (lengths -> offsets by a multi-level block
// scan, then either one thread per run or a binary search per output byte),
// CRC32 of 4 KiB chunks (table in __constant__ and table copied to shared
// memory) and base64 encoding (table in __constant__, both padding forms).
#include "stress.cuh"

// ---------------------------------------------------------------------------
// Pattern count
// ---------------------------------------------------------------------------

__constant__ unsigned char c_pat[8];

constexpr int kCountThreads = 256;
constexpr int kCountPerThread = 16;
constexpr int kCountTile = kCountThreads * kCountPerThread;  // match start positions per block

// Each block owns kCountTile start positions; it stages them plus a halo of
// PLEN-1 bytes in shared memory, so matches crossing the tile end are found.
template <int PLEN>
__global__ void __launch_bounds__(kCountThreads) count_pattern(const unsigned char* __restrict__ buf, size_t n,
                                                               unsigned long long* total) {
    __shared__ __align__(16) unsigned char tile[kCountTile + 16];
    __shared__ unsigned warp_sums[kCountThreads / 32];
    const size_t base = size_t(blockIdx.x) * kCountTile;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // Body: one 16-byte vector per thread when the tile is fully in range.
    size_t off = base + size_t(tid) * 16;
    if (off + 16 <= n) {
        reinterpret_cast<uint4*>(tile)[tid] = *reinterpret_cast<const uint4*>(buf + off);
    } else {
        for (int q = 0; q < 16; ++q) tile[tid * 16 + q] = off + q < n ? buf[off + q] : 0;
    }
    if (tid < PLEN - 1) {
        size_t h = base + kCountTile + tid;
        tile[kCountTile + tid] = h < n ? buf[h] : 0;
    }
    __syncthreads();

    unsigned count = 0;
    const size_t last_start = n >= PLEN ? n - PLEN : 0;
#pragma unroll 4
    for (int q = 0; q < kCountPerThread; ++q) {
        int p = q * kCountThreads + tid;  // consecutive lanes, consecutive bytes
        bool hit = n >= PLEN && base + p <= last_start;
#pragma unroll
        for (int k = 0; k < PLEN; ++k) hit = hit && tile[p + k] == c_pat[k];
        unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) count += __popc(m);
    }
    if (lane == 0) warp_sums[warp] = count;
    __syncthreads();
    if (warp == 0) {
        unsigned v = lane < kCountThreads / 32 ? warp_sums[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0 && v) atomicAdd(total, (unsigned long long)v);
    }
}

static unsigned long long host_count(const std::vector<unsigned char>& buf, const unsigned char* pat, int plen) {
    const long long last = (long long)buf.size() - plen;
    unsigned long long c = 0;
#pragma omp parallel for reduction(+ : c) schedule(static)
    for (long long i = 0; i <= last; ++i) {
        if (buf[i] != pat[0]) continue;
        int k = 1;
        while (k < plen && buf[i + k] == pat[k]) ++k;
        c += k == plen;
    }
    return c;
}

template <int PLEN>
static void count_case(const char* name, const std::vector<unsigned char>& host, const unsigned char* d_buf,
                       const char* pat) {
    unsigned char p[8];
    std::memcpy(p, pat, PLEN);
    unsigned long long want = 0;
    double ref = stress::best_ms([&] { want = host_count(host, p, PLEN); });
    STRESS_CHECK(cudaMemcpyToSymbol(c_pat, p, PLEN));
    stress::Buffer<unsigned long long> d_total(1);
    const size_t n = host.size();
    const unsigned blocks = unsigned((n + kCountTile - 1) / kCountTile);
    double k = stress::time_device([&] {
        STRESS_CHECK(cudaMemset(d_total.p, 0, sizeof(unsigned long long)));
        count_pattern<PLEN><<<blocks, kCountThreads>>>(d_buf, n, d_total);
    });
    unsigned long long got = d_total.download()[0];
    if (got != want) std::printf("  %s: got %llu want %llu\n", name, got, want);
    char note[64];
    std::snprintf(note, sizeof note, "matches=%llu", want);
    stress::report(name, got == want && want > 0, k, ref, note);
}

// ---------------------------------------------------------------------------
// Run-length decode
// ---------------------------------------------------------------------------

// Exclusive scan of a block's 4*blockDim.x values: each thread scans its four
// in registers, warps scan their thread totals with shuffles, warp 0 scans the
// warp totals.
__device__ uint4 block_exclusive_scan4(uint4 v, unsigned* block_total) {
    __shared__ unsigned warp_tot[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = (blockDim.x + 31) >> 5;
    unsigned s0 = v.x, s1 = s0 + v.y, s2 = s1 + v.z, t = s2 + v.w;
    unsigned inc = t;
    for (int o = 1; o < 32; o <<= 1) {
        unsigned y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        unsigned w = lane < nwarps ? warp_tot[lane] : 0, wi = w;
        for (int o = 1; o < 32; o <<= 1) {
            unsigned y = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= o) wi += y;
        }
        if (lane < nwarps) warp_tot[lane] = wi - w;  // exclusive
        if (lane == nwarps - 1) *block_total = wi;
    }
    __syncthreads();
    unsigned ex = warp_tot[warp] + inc - t;
    return make_uint4(ex, ex + s0, ex + s1, ex + s2);
}

// Pass 1 (and the single-block pass over the block sums): in-place-capable
// exclusive scan of n values, 4 per thread, writes each block's total.
__global__ void scan_blocks(const unsigned* in, unsigned* out, unsigned* block_sums, unsigned n) {
    __shared__ unsigned total;
    unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (i + 3 < n) {
        v = *reinterpret_cast<const uint4*>(in + i);
    } else {
        if (i < n) v.x = in[i];
        if (i + 1 < n) v.y = in[i + 1];
        if (i + 2 < n) v.z = in[i + 2];
    }
    uint4 r = block_exclusive_scan4(v, &total);
    if (i + 3 < n) {
        *reinterpret_cast<uint4*>(out + i) = r;
    } else {
        if (i < n) out[i] = r.x;
        if (i + 1 < n) out[i + 1] = r.y;
        if (i + 2 < n) out[i + 2] = r.z;
    }
    if (threadIdx.x == 0 && block_sums) block_sums[blockIdx.x] = total;
}

__global__ void add_block_offsets(unsigned* data, const unsigned* block_offsets, unsigned n) {
    unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    unsigned add = block_offsets[blockIdx.x];
    if (i + 3 < n) {
        uint4 v = *reinterpret_cast<uint4*>(data + i);
        v.x += add, v.y += add, v.z += add, v.w += add;
        *reinterpret_cast<uint4*>(data + i) = v;
    } else {
        for (unsigned k = i; k < n; ++k) data[k] += add;
    }
}

__global__ void rle_scatter(const unsigned char* values, const unsigned* lengths, const unsigned* offsets,
                            unsigned char* out, unsigned nruns) {
    unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nruns) return;
    unsigned char v = values[r];
    unsigned o = offsets[r], len = lengths[r];
    for (unsigned k = 0; k < len; ++k) out[o + k] = v;
}

// One thread per output byte: the last run whose offset is <= the position.
__global__ void rle_bsearch(const unsigned char* values, const unsigned* offsets, unsigned nruns, unsigned char* out,
                            unsigned total) {
    unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total) return;
    unsigned lo = 0, hi = nruns;  // offsets[lo] <= p < offsets[hi]
    while (hi - lo > 1) {
        unsigned mid = (lo + hi) >> 1;
        if (offsets[mid] <= p) lo = mid;
        else hi = mid;
    }
    out[p] = values[lo];
}

constexpr int kScanThreads = 256, kScanItems = kScanThreads * 4;

static void device_scan(const unsigned* in, unsigned* out, unsigned* sums, unsigned n) {
    unsigned blocks = (n + kScanItems - 1) / kScanItems;  // <= 4096: the sums fit in one 1024-thread block
    scan_blocks<<<blocks, kScanThreads>>>(in, out, sums, n);
    scan_blocks<<<1, 1024>>>(sums, sums, nullptr, blocks);
    add_block_offsets<<<blocks, kScanThreads>>>(out, sums, n);
}

static void rle_cases() {
    using namespace stress;
    const unsigned nruns = 1u << 22;
    std::vector<unsigned char> values(nruns);
    std::vector<unsigned> lengths(nruns);
    for (unsigned r = 0; r < nruns; ++r) {
        uint32_t h = hash32(r * 0x9e3779b9u + 7);
        values[r] = (unsigned char)h;
        lengths[r] = 1 + ((h >> 8) & 15);
    }
    // Host reference: blocked parallel exclusive scan, then a parallel fill.
    std::vector<unsigned> want_off(nruns);
    std::vector<unsigned char> want;
    unsigned total = 0;
    auto host_decode = [&] {
        int nt = omp_get_max_threads();
        std::vector<unsigned> part(nt + 1, 0);
#pragma omp parallel num_threads(nt)
        {
            int t = omp_get_thread_num();
            unsigned lo = unsigned(uint64_t(nruns) * t / nt), hi = unsigned(uint64_t(nruns) * (t + 1) / nt);
            unsigned s = 0;
            for (unsigned r = lo; r < hi; ++r) s += lengths[r];
            part[t + 1] = s;
#pragma omp barrier
#pragma omp single
            for (int k = 0; k < nt; ++k) part[k + 1] += part[k];
            s = part[t];
            for (unsigned r = lo; r < hi; ++r) want_off[r] = s, s += lengths[r];
        }
        total = part[nt];
        want.resize(total);
#pragma omp parallel for schedule(static)
        for (long long r = 0; r < (long long)nruns; ++r) std::memset(&want[want_off[r]], values[r], lengths[r]);
    };
    double ref = best_ms(host_decode);

    Buffer<unsigned char> d_values(values), d_out(total);
    Buffer<unsigned> d_len(lengths), d_off(nruns), d_sums(4096);
    double k1 = time_device([&] {
        device_scan(d_len, d_off, d_sums, nruns);
        rle_scatter<<<(nruns + 255) / 256, 256>>>(d_values, d_len, d_off, d_out, nruns);
    });
    auto got_off = d_off.download();
    auto got = d_out.download();
    bool ok = equal(got_off.data(), want_off.data(), nruns, "offset") && equal(got.data(), want.data(), total, "out");
    report("rle_decode_scatter", ok, k1, ref);

    STRESS_CHECK(cudaMemset(d_out.p, 0, total));
    STRESS_CHECK(cudaMemset(d_off.p, 0, nruns * sizeof(unsigned)));
    double k2 = time_device([&] {
        device_scan(d_len, d_off, d_sums, nruns);
        rle_bsearch<<<(total + 255) / 256, 256>>>(d_values, d_off, nruns, d_out, total);
    });
    got_off = d_off.download();
    got = d_out.download();
    ok = equal(got_off.data(), want_off.data(), nruns, "offset") && equal(got.data(), want.data(), total, "out");
    report("rle_decode_bsearch", ok, k2, ref);
}

// ---------------------------------------------------------------------------
// CRC32 (IEEE, reflected polynomial 0xEDB88320) per chunk
// ---------------------------------------------------------------------------

constexpr int kChunk = 4096;
__constant__ uint32_t c_crc_table[256];

__device__ __forceinline__ uint32_t crc_word(uint32_t crc, uint32_t w, const uint32_t* table) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        crc = table[(crc ^ w) & 0xff] ^ (crc >> 8);
        w >>= 8;
    }
    return crc;
}

__device__ __forceinline__ uint32_t crc_chunk(const unsigned char* data, const uint32_t* table) {
    const uint4* p = reinterpret_cast<const uint4*>(data);
    uint32_t crc = 0xffffffffu;
    for (int i = 0; i < kChunk / 16; ++i) {
        uint4 v = p[i];
        crc = crc_word(crc, v.x, table);
        crc = crc_word(crc, v.y, table);
        crc = crc_word(crc, v.z, table);
        crc = crc_word(crc, v.w, table);
    }
    return ~crc;
}

__global__ void crc32_chunks_const(const unsigned char* __restrict__ data, uint32_t* __restrict__ out, int nchunks) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nchunks) out[c] = crc_chunk(data + size_t(c) * kChunk, c_crc_table);
}

__global__ void crc32_chunks_shared(const unsigned char* __restrict__ data, const uint32_t* __restrict__ table,
                                    uint32_t* __restrict__ out, int nchunks) {
    __shared__ uint32_t s_table[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_table[i] = table[i];
    __syncthreads();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nchunks) out[c] = crc_chunk(data + size_t(c) * kChunk, s_table);
}

static void crc_cases(const std::vector<unsigned char>& host, const unsigned char* d_buf) {
    using namespace stress;
    // Slicing-by-8 tables for the host reference; table 0 is the classic one.
    static uint32_t T[8][256];
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
        T[0][i] = c;
    }
    for (int s = 1; s < 8; ++s)
        for (int i = 0; i < 256; ++i) T[s][i] = (T[s - 1][i] >> 8) ^ T[0][T[s - 1][i] & 0xff];
    const int nchunks = int(host.size() / kChunk);
    std::vector<uint32_t> want(nchunks);
    double ref = best_ms([&] {
#pragma omp parallel for schedule(static)
        for (int c = 0; c < nchunks; ++c) {
            const unsigned char* p = host.data() + size_t(c) * kChunk;
            uint32_t crc = 0xffffffffu;
            for (int i = 0; i < kChunk; i += 8) {
                uint32_t a, b;
                std::memcpy(&a, p + i, 4);
                std::memcpy(&b, p + i + 4, 4);
                a ^= crc;
                crc = T[7][a & 0xff] ^ T[6][(a >> 8) & 0xff] ^ T[5][(a >> 16) & 0xff] ^ T[4][a >> 24] ^
                      T[3][b & 0xff] ^ T[2][(b >> 8) & 0xff] ^ T[1][(b >> 16) & 0xff] ^ T[0][b >> 24];
            }
            want[c] = ~crc;
        }
    });
    STRESS_CHECK(cudaMemcpyToSymbol(c_crc_table, T[0], sizeof(T[0])));
    Buffer<uint32_t> d_table(std::vector<uint32_t>(T[0], T[0] + 256)), d_out(nchunks);
    const int threads = 128, blocks = (nchunks + threads - 1) / threads;

    d_out.zero();
    double k1 = time_device([&] { crc32_chunks_const<<<blocks, threads>>>(d_buf, d_out, nchunks); });
    auto got = d_out.download();
    report("crc32_const_table", equal(got.data(), want.data(), nchunks, "crc"), k1, ref);

    d_out.zero();
    double k2 = time_device([&] { crc32_chunks_shared<<<blocks, threads>>>(d_buf, d_table, d_out, nchunks); });
    got = d_out.download();
    report("crc32_shared_table", equal(got.data(), want.data(), nchunks, "crc"), k2, ref);
}

// ---------------------------------------------------------------------------
// Base64 encode
// ---------------------------------------------------------------------------

__constant__ char c_b64[64];
static const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// One thread per 3-byte group, four output characters stored as one uchar4.
__global__ void base64_encode(const unsigned char* __restrict__ in, size_t n, uchar4* __restrict__ out, size_t groups) {
    size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    size_t i = g * 3;
    unsigned b0 = in[i];
    unsigned b1 = i + 1 < n ? in[i + 1] : 0;
    unsigned b2 = i + 2 < n ? in[i + 2] : 0;
    unsigned w = (b0 << 16) | (b1 << 8) | b2;
    uchar4 o;
    o.x = c_b64[(w >> 18) & 63];
    o.y = c_b64[(w >> 12) & 63];
    o.z = i + 1 < n ? c_b64[(w >> 6) & 63] : '=';
    o.w = i + 2 < n ? c_b64[w & 63] : '=';
    out[g] = o;
}

static void base64_case(const char* name, const std::vector<unsigned char>& host, const unsigned char* d_buf, size_t n) {
    using namespace stress;
    const size_t groups = (n + 2) / 3;
    std::vector<unsigned char> want(groups * 4);
    double ref = best_ms([&] {
#pragma omp parallel for simd schedule(static)
        for (long long g = 0; g < (long long)groups; ++g) {
            size_t i = size_t(g) * 3;
            unsigned b0 = host[i], b1 = i + 1 < n ? host[i + 1] : 0, b2 = i + 2 < n ? host[i + 2] : 0;
            unsigned w = (b0 << 16) | (b1 << 8) | b2;
            unsigned char* o = &want[size_t(g) * 4];
            o[0] = kB64[(w >> 18) & 63];
            o[1] = kB64[(w >> 12) & 63];
            o[2] = i + 1 < n ? kB64[(w >> 6) & 63] : '=';
            o[3] = i + 2 < n ? kB64[w & 63] : '=';
        }
    });
    Buffer<uchar4> d_out(groups);
    d_out.zero();
    double k = time_device([&] { base64_encode<<<unsigned((groups + 255) / 256), 256>>>(d_buf, n, d_out, groups); });
    std::vector<uchar4> got4 = d_out.download();
    const unsigned char* got = reinterpret_cast<const unsigned char*>(got4.data());
    report(name, equal(got, want.data(), groups * 4, "b64"), k, ref);
}

int main() {
    using namespace stress;
    const size_t n = size_t(64) << 20;
    // Random bytes from a small alphabet, so the patterns match often.
    std::vector<unsigned char> acgt(n), ab(n);
#pragma omp parallel for
    for (long long i = 0; i < (long long)n; ++i) {
        uint32_t h = hash32(uint32_t(i) * 2654435761u + 12345u);
        acgt[i] = "ACGT"[h & 3];
        ab[i] = "ab"[(h >> 7) & 1];
    }
    {
        Buffer<unsigned char> d(acgt);
        count_case<6>("count_pattern6_acgt", acgt, d, "GATTAC");
        Buffer<unsigned char> d2(ab);
        count_case<8>("count_pattern8_overlapping", ab, d2, "abababab");
    }

    rle_cases();

    std::vector<unsigned char> bytes(n);
    auto words = random_u32(n / 4, 99);
    std::memcpy(bytes.data(), words.data(), n);
    {
        Buffer<unsigned char> d(bytes);
        crc_cases(bytes, d);
        STRESS_CHECK(cudaMemcpyToSymbol(c_b64, kB64, 64));
        base64_case("base64_pad1", bytes, d, (size_t(48) << 20) + 2);  // n % 3 == 2: one '='
        base64_case("base64_pad2", bytes, d, (size_t(12) << 20) + 1);  // n % 3 == 1: "=="
    }
    return finish();
}
