// Launch geometry, 2D and 3D: block and grid shapes in y and z, gridDim.y =
// 65535, the same kernel relaunched with a different shape every time, and
// 2D/3D stencils with partial blocks on every edge.
#include "stress.cuh"

struct Id {
    unsigned gid, tid, bx, byz;
    bool operator==(const Id& o) const { return gid == o.gid && tid == o.tid && bx == o.bx && byz == o.byz; }
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

// 2D 5-point stencil on a W x H image (neither a multiple of the block).
__host__ __device__ inline float at(const float* img, int w, int h, int x, int y) {
    x = x < 0 ? 0 : x >= w ? w - 1 : x;
    y = y < 0 ? 0 : y >= h ? h - 1 : y;
    return img[y * w + x];
}
__host__ __device__ inline float stencil2d(const float* img, int w, int h, int x, int y) {
    float c = at(img, w, h, x, y);
    float s = at(img, w, h, x - 1, y) + at(img, w, h, x + 1, y) + at(img, w, h, x, y - 1) + at(img, w, h, x, y + 1);
    return c * 0.5f + s * 0.125f;
}
__global__ void blur2d(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out[y * w + x] = stencil2d(in, w, h, x, y);
}

// 3D: each voxel gets the sum of its coordinates weighted, plus its 6 neighbours.
__host__ __device__ inline int vox(const int* v, int nx, int ny, int nz, int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
    return v[(z * ny + y) * nx + x];
}
__host__ __device__ inline int stencil3d(const int* v, int nx, int ny, int nz, int x, int y, int z) {
    return 6 * vox(v, nx, ny, nz, x, y, z) - vox(v, nx, ny, nz, x - 1, y, z) - vox(v, nx, ny, nz, x + 1, y, z) -
           vox(v, nx, ny, nz, x, y - 1, z) - vox(v, nx, ny, nz, x, y + 1, z) - vox(v, nx, ny, nz, x, y, z - 1) -
           vox(v, nx, ny, nz, x, y, z + 1);
}
__global__ void lap3d(const int* in, int* out, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) out[(z * ny + y) * nx + x] = stencil3d(in, nx, ny, nz, x, y, z);
}

int main() {
    using namespace stress;

    // 2D blocks.
    identity_case("block_16x16", dim3(13, 11), dim3(16, 16));
    identity_case("block_32x8", dim3(9, 17), dim3(32, 8));
    identity_case("block_8x32", dim3(17, 9), dim3(8, 32));
    identity_case("block_3x5", dim3(101, 37), dim3(3, 5));
    identity_case("block_32x32", dim3(7, 5), dim3(32, 32));
    // 3D blocks.
    identity_case("block_8x8x8", dim3(5, 4, 3), dim3(8, 8, 8));
    identity_case("block_4x4x64", dim3(3, 5, 7), dim3(4, 4, 64));
    identity_case("block_2x3x5", dim3(31, 7, 11), dim3(2, 3, 5));
    identity_case("block_1x1x64", dim3(9, 3, 4), dim3(1, 1, 64));
    identity_case("block_1x1024", dim3(3, 2), dim3(1, 1024));
    // 2D and 3D grids.
    identity_case("grid_y_65535", dim3(2, 65535), dim3(32));
    identity_case("grid_z_64", dim3(7, 9, 64), dim3(33));
    identity_case("grid_z_65535", dim3(1, 1, 65535), dim3(32));
    identity_case("grid_3d_block_3d", dim3(6, 5, 4), dim3(5, 6, 7));
    identity_case("grid_y_1000_block_1", dim3(3, 1000, 2), dim3(1));

    // The same kernel with a new shape on every launch, all into one buffer
    // region per launch, checked after all launches.
    {
        const dim3 shapes[][2] = {
            {dim3(10), dim3(1024)},       {dim3(37), dim3(7)},          {dim3(4, 4), dim3(16, 16)},
            {dim3(3, 3, 3), dim3(8, 8, 8)}, {dim3(100), dim3(33)},        {dim3(2, 7), dim3(3, 5)},
            {dim3(1), dim3(1)},           {dim3(5, 1, 6), dim3(2, 3, 5)}, {dim3(64), dim3(31)},
            {dim3(1, 9), dim3(32, 32)},   {dim3(11, 2, 3), dim3(4, 4, 64)}, {dim3(1000), dim3(1)},
        };
        const int ns = sizeof shapes / sizeof shapes[0];
        std::vector<size_t> off(ns + 1, 0);
        for (int s = 0; s < ns; ++s) {
            const dim3 &g = shapes[s][0], &b = shapes[s][1];
            off[s + 1] = off[s] + size_t(g.x) * g.y * g.z * b.x * b.y * b.z;
        }
        std::vector<Id> want(off[ns]);
        double ref = best_ms([&] {
            for (int s = 0; s < ns; ++s) {
                auto v = expected_ids(shapes[s][0], shapes[s][1]);
                std::copy(v.begin(), v.end(), want.begin() + off[s]);
            }
        });
        Buffer<Id> d(off[ns]);
        double k = time_device([&] {
            for (int s = 0; s < ns; ++s) identity<<<shapes[s][0], shapes[s][1]>>>(d.p + off[s]);
        });
        report("reshape_12_launches", check_ids(d.download(), want), k, ref);
    }

    // 2D stencil with several block shapes; image size not a multiple of any.
    {
        const int w = 1023, h = 517;
        auto img = random_floats(size_t(w) * h, 5);
        std::vector<float> want(size_t(w) * h);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) want[size_t(y) * w + x] = stencil2d(img.data(), w, h, x, y);
        });
        Buffer<float> din(img), dout(size_t(w) * h);
        const dim3 blocks[] = {dim3(16, 16), dim3(32, 8), dim3(8, 32), dim3(3, 5), dim3(32, 32)};
        const char* names[] = {"blur2d_16x16", "blur2d_32x8", "blur2d_8x32", "blur2d_3x5", "blur2d_32x32"};
        for (int i = 0; i < 5; ++i) {
            dout.zero();
            dim3 b = blocks[i], g((w + b.x - 1) / b.x, (h + b.y - 1) / b.y);
            double k = time_device([&] { blur2d<<<g, b>>>(din, dout, w, h); });
            // Same operations in the same order on both sides; tolerance only
            // for a possible fma contraction of c*0.5f + s*0.125f.
            report(names[i], close(dout.download().data(), want.data(), want.size(), 1e-6, 1e-7), k, ref);
        }
    }

    // 3D Laplacian, integer (exact), several 3D block shapes.
    {
        const int nx = 67, ny = 45, nz = 39;
        size_t n = size_t(nx) * ny * nz;
        auto r = random_u32(n, 9);
        std::vector<int> v(n), want(n);
        for (size_t i = 0; i < n; ++i) v[i] = int(r[i] % 2001) - 1000;
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x) want[(size_t(z) * ny + y) * nx + x] = stencil3d(v.data(), nx, ny, nz, x, y, z);
        });
        Buffer<int> din(v), dout(n);
        const dim3 blocks[] = {dim3(8, 8, 8), dim3(4, 4, 64), dim3(2, 3, 5), dim3(32, 4, 2)};
        const char* names[] = {"lap3d_8x8x8", "lap3d_4x4x64", "lap3d_2x3x5", "lap3d_32x4x2"};
        for (int i = 0; i < 4; ++i) {
            dout.zero();
            dim3 b = blocks[i], g((nx + b.x - 1) / b.x, (ny + b.y - 1) / b.y, (nz + b.z - 1) / b.z);
            double k = time_device([&] { lap3d<<<g, b>>>(din, dout, nx, ny, nz); });
            report(names[i], equal(dout.download().data(), want.data(), n), k, ref);
        }
    }
    return finish();
}
