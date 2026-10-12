// Launch geometry, 1D: odd block sizes, partial last blocks with an early
// return, a single block, a grid of 2^20 blocks of 32 threads.
#include "stress.cuh"

// Every thread writes its full identity at its own global linear index, so a
// wrong threadIdx/blockIdx/blockDim/gridDim anywhere shows up as a mismatch.
struct Id {
    unsigned gid, tid, bx, byz;
    bool operator==(const Id& o) const { return gid == o.gid && tid == o.tid && bx == o.bx && byz == o.byz; }
    explicit operator double() const { return double(gid) * 4294967296.0 + tid; }
};

__global__ void identity(Id* out) {
    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    unsigned block = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    unsigned t = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned gid = block * threads + t;
    out[gid] = Id{gid, threadIdx.x | (threadIdx.y << 11) | (threadIdx.z << 22), blockIdx.x,
                  blockIdx.y | (blockIdx.z << 16)};
}

static std::vector<Id> expected_ids(dim3 g, dim3 b) {
    size_t threads = size_t(b.x) * b.y * b.z, blocks = size_t(g.x) * g.y * g.z;
    std::vector<Id> v(threads * blocks);
#pragma omp parallel for
    for (long long blk = 0; blk < (long long)blocks; ++blk) {
        unsigned bx = blk % g.x, by = (blk / g.x) % g.y, bz = blk / (size_t(g.x) * g.y);
        for (size_t t = 0; t < threads; ++t) {
            unsigned tx = t % b.x, ty = (t / b.x) % b.y, tz = t / (size_t(b.x) * b.y);
            size_t gid = blk * threads + t;
            v[gid] = Id{unsigned(gid), tx | (ty << 11) | (tz << 22), bx, by | (bz << 16)};
        }
    }
    return v;
}

static bool check_ids(const std::vector<Id>& got, const std::vector<Id>& want) {
    for (size_t i = 0; i < want.size(); ++i)
        if (!(got[i] == want[i])) {
            std::printf("  mismatch [%zu]: got gid=%u tid=%x bx=%u byz=%x want gid=%u tid=%x bx=%u byz=%x\n", i, got[i].gid,
                        got[i].tid, got[i].bx, got[i].byz, want[i].gid, want[i].tid, want[i].bx, want[i].byz);
            return false;
        }
    return true;
}

static void identity_case(const char* name, dim3 g, dim3 b) {
    std::vector<Id> want;
    double ref = stress::best_ms([&] { want = expected_ids(g, b); });
    stress::Buffer<Id> d(want.size());
    STRESS_CHECK(cudaMemset(d.p, 0xff, want.size() * sizeof(Id)));
    double k = stress::time_device([&] { identity<<<g, b>>>(d); });
    stress::report(name, check_ids(d.download(), want), k, ref);
}

// Partial last block: threads past n return early, before any memory access.
__global__ void scale_tail(const float* x, float* y, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a * x[i] + float(i % 7);
}

// Grid-stride loop with a grid much smaller than the data.
__global__ void grid_stride(const unsigned* x, unsigned* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) y[i] = x[i] * 2654435761u + i;
}

// 2^20 blocks of 32: one word per thread (the hash of its linear index).
__host__ __device__ inline unsigned mix(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x;
}
__global__ void many_blocks(unsigned* out) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = mix(gid) ^ (blockIdx.x << 5);
}

int main() {
    using namespace stress;
    const int sizes[] = {1, 7, 31, 33, 63, 100, 255, 257, 1000, 1024};

    // Identity with each 1D block size (grid sized so the total stays moderate).
    for (int bs : sizes) {
        char name[64];
        std::snprintf(name, sizeof name, "identity_block_%d", bs);
        identity_case(name, dim3((200000 + bs - 1) / bs), dim3(bs));
    }

    // Early return in partial last blocks, for every block size.
    {
        const int n = 100003;
        auto x = random_floats(n, 11);
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = 1.5f * x[i] + float(i % 7);
        });
        Buffer<float> dx(x), dy(n);
        for (int bs : sizes) {
            dy.zero();
            int grid = (n + bs - 1) / bs;
            double k = time_device([&] { scale_tail<<<grid, bs>>>(dx, dy, n, 1.5f); });
            char name[64];
            std::snprintf(name, sizeof name, "early_return_block_%d", bs);
            report(name, equal(dy.download().data(), want.data(), n), k, ref);
        }
    }

    // Single block, every thread of 1024 and of 1.
    identity_case("single_block_1024", dim3(1), dim3(1024));
    identity_case("single_thread", dim3(1), dim3(1));

    // Grid-stride loop: 3 blocks of 96 over 1M elements.
    {
        const int n = 1 << 20;
        auto x = random_u32(n, 3);
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) want[i] = x[i] * 2654435761u + i;
        });
        Buffer<unsigned> dx(x), dy(n);
        double k = time_device([&] { grid_stride<<<3, 96>>>(dx, dy, n); });
        report("grid_stride_3x96", equal(dy.download().data(), want.data(), n), k, ref);
    }

    // 2^20 blocks of 32 threads.
    {
        const unsigned blocks = 1u << 20, n = blocks * 32;
        std::vector<unsigned> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (long long i = 0; i < n; ++i) want[i] = mix(unsigned(i)) ^ ((unsigned(i) >> 5) << 5);
        });
        Buffer<unsigned> d(n);
        double k = time_device([&] { many_blocks<<<blocks, 32>>>(d); });
        report("grid_2pow20_blocks_of_32", equal(d.download().data(), want.data(), n), k, ref);
    }

    // A wide 1D grid (more than 65535 blocks in x) with full identity.
    identity_case("identity_grid_x_100000", dim3(100000), dim3(33));
    return finish();
}
