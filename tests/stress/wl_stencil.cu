// Stencil workloads: 1D 7-point stencil with shared-memory halos, 2D 5-point
// Jacobi iterations (ping-pong launches), 3D 7-point stencil marching in z
// with a shared xy plane, 2D 5x5 convolution with the filter in __constant__
// and a shared tile, separable Gaussian blur (row and column passes, filter in
// __constant__), and Sobel edge detection on 8-bit images.
#include "stress.cuh"

// ---------------------------------------------------------------- 1D stencil
#define RADIUS 3
#define BLOCK1D 256

__constant__ float c_w1d[2 * RADIUS + 1];

__global__ void stencil1d(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float temp[BLOCK1D + 2 * RADIUS];
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int l = threadIdx.x + RADIUS;
    temp[l] = g < n ? in[g] : 0.0f;
    if (threadIdx.x < RADIUS) {
        int gl = g - RADIUS, gr = g + BLOCK1D;
        temp[l - RADIUS] = gl >= 0 && gl < n ? in[gl] : 0.0f;
        temp[l + BLOCK1D] = gr < n ? in[gr] : 0.0f;
    }
    __syncthreads();
    if (g < n) {
        float s = 0.0f;
#pragma unroll
        for (int o = -RADIUS; o <= RADIUS; ++o) s += c_w1d[o + RADIUS] * temp[l + o];
        out[g] = s;
    }
}

// ------------------------------------------------------------- 2D Jacobi
__global__ void jacobi2d(const float* __restrict__ in, float* __restrict__ out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int i = y * nx + x;
    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1)
        out[i] = in[i];
    else
        out[i] = 0.25f * (in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx]);
}

// ------------------------------------------------------------- 3D stencil
#define BX3 32
#define BY3 8

// Each thread owns one (x, y) column and marches in z, keeping the points
// behind and in front in registers and the current plane in shared memory.
__global__ void stencil3d(const float* __restrict__ in, float* __restrict__ out, int nx, int ny, int nz, float c0,
                          float c1) {
    __shared__ float plane[BY3 + 2][BX3 + 2];
    int x = blockIdx.x * BX3 + threadIdx.x;
    int y = blockIdx.y * BY3 + threadIdx.y;
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
    bool inside = x < nx && y < ny;
    bool edge = x == 0 || y == 0 || x == nx - 1 || y == ny - 1;
    size_t stride = size_t(nx) * ny;
    size_t col = size_t(y) * nx + x;

    float behind = 0.0f, cur = 0.0f, front = 0.0f;
    if (inside) {
        cur = in[col];
        front = in[col + stride];
        out[col] = cur;
    }
    for (int z = 1; z < nz - 1; ++z) {
        behind = cur;
        cur = front;
        size_t i = col + z * stride;
        if (inside) front = in[i + stride];
        __syncthreads();
        plane[ty][tx] = cur;
        if (inside) {
            if (threadIdx.x == 0 && x > 0) plane[ty][0] = in[i - 1];
            if (threadIdx.x == BX3 - 1 && x < nx - 1) plane[ty][BX3 + 1] = in[i + 1];
            if (threadIdx.y == 0 && y > 0) plane[0][tx] = in[i - nx];
            if (threadIdx.y == BY3 - 1 && y < ny - 1) plane[BY3 + 1][tx] = in[i + nx];
        }
        __syncthreads();
        if (inside) {
            if (edge)
                out[i] = cur;
            else
                out[i] = c0 * cur + c1 * (behind + front + plane[ty][tx - 1] + plane[ty][tx + 1] + plane[ty - 1][tx] +
                                          plane[ty + 1][tx]);
        }
    }
    if (inside) out[col + (nz - 1) * stride] = front;
}

// ------------------------------------------------------- 2D convolution 5x5
#define FR 2
#define FW (2 * FR + 1)
#define CTILE 16

__constant__ float c_filter[FW * FW];

__global__ void conv2d(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
    __shared__ float tile[CTILE + 2 * FR][CTILE + 2 * FR];
    int x0 = blockIdx.x * CTILE, y0 = blockIdx.y * CTILE;
    for (int j = threadIdx.y; j < CTILE + 2 * FR; j += CTILE)
        for (int i = threadIdx.x; i < CTILE + 2 * FR; i += CTILE) {
            int gx = x0 + i - FR, gy = y0 + j - FR;
            tile[j][i] = gx >= 0 && gx < w && gy >= 0 && gy < h ? in[gy * w + gx] : 0.0f;
        }
    __syncthreads();
    int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x < w && y < h) {
        float s = 0.0f;
#pragma unroll
        for (int ky = 0; ky < FW; ++ky)
#pragma unroll
            for (int kx = 0; kx < FW; ++kx) s += c_filter[ky * FW + kx] * tile[threadIdx.y + ky][threadIdx.x + kx];
        out[y * w + x] = s;
    }
}

// --------------------------------------------------- separable Gaussian blur
#define GR 8
#define GW (2 * GR + 1)
#define GBX 32
#define GBY 8

__constant__ float c_gauss[GW];

// Row pass: each block loads its rows plus GR pixels of halo on each side
// (clamped to the edge).
__global__ void gauss_rows(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
    __shared__ float tile[GBY][GBX + 2 * GR];
    int x0 = blockIdx.x * GBX;
    int y = blockIdx.y * GBY + threadIdx.y;
    int yc = min(y, h - 1);
    for (int i = threadIdx.x; i < GBX + 2 * GR; i += GBX) {
        int gx = min(max(x0 + i - GR, 0), w - 1);
        tile[threadIdx.y][i] = in[yc * w + gx];
    }
    __syncthreads();
    int x = x0 + threadIdx.x;
    if (x < w && y < h) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < GW; ++k) s += c_gauss[k] * tile[threadIdx.y][threadIdx.x + k];
        out[y * w + x] = s;
    }
}

// Column pass: the halo is above and below.
__global__ void gauss_cols(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
    __shared__ float tile[GBY + 2 * GR][GBX];
    int x = blockIdx.x * GBX + threadIdx.x;
    int y0 = blockIdx.y * GBY;
    int xc = min(x, w - 1);
    for (int j = threadIdx.y; j < GBY + 2 * GR; j += GBY) {
        int gy = min(max(y0 + j - GR, 0), h - 1);
        tile[j][threadIdx.x] = in[gy * w + xc];
    }
    __syncthreads();
    int y = y0 + threadIdx.y;
    if (x < w && y < h) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < GW; ++k) s += c_gauss[k] * tile[threadIdx.y + k][threadIdx.x];
        out[y * w + x] = s;
    }
}

// ---------------------------------------------------------------- Sobel
#define SBX 32
#define SBY 8

__global__ void sobel(const unsigned char* __restrict__ in, unsigned char* __restrict__ out, int w, int h) {
    __shared__ unsigned char tile[SBY + 2][SBX + 2];
    int x0 = blockIdx.x * SBX, y0 = blockIdx.y * SBY;
    for (int j = threadIdx.y; j < SBY + 2; j += SBY)
        for (int i = threadIdx.x; i < SBX + 2; i += SBX) {
            int gx = x0 + i - 1, gy = y0 + j - 1;
            tile[j][i] = gx >= 0 && gx < w && gy >= 0 && gy < h ? in[gy * w + gx] : 0;
        }
    __syncthreads();
    int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    if (x >= w || y >= h) return;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * w + x] = 0;
        return;
    }
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
    int gx = (tile[ty - 1][tx + 1] + 2 * tile[ty][tx + 1] + tile[ty + 1][tx + 1]) -
             (tile[ty - 1][tx - 1] + 2 * tile[ty][tx - 1] + tile[ty + 1][tx - 1]);
    int gy = (tile[ty + 1][tx - 1] + 2 * tile[ty + 1][tx] + tile[ty + 1][tx + 1]) -
             (tile[ty - 1][tx - 1] + 2 * tile[ty - 1][tx] + tile[ty - 1][tx + 1]);
    float mag = sqrtf(float(gx * gx + gy * gy));
    out[y * w + x] = (unsigned char)min(255, int(mag));
}

// ===================================================================== cases
using namespace stress;

static void case_stencil1d() {
    const int n = (1 << 25) + 17;
    auto in = random_floats(n, 11);
    float w[2 * RADIUS + 1] = {0.05f, -0.1f, 0.2f, 0.5f, 0.2f, -0.1f, 0.05f};
    STRESS_CHECK(cudaMemcpyToSymbol(c_w1d, w, sizeof(w)));
    std::vector<float> want(n);
    double ref = best_ms([&] {
#pragma omp parallel for simd
        for (int i = 0; i < n; ++i) {
            float s = 0.0f;
            for (int o = -RADIUS; o <= RADIUS; ++o) {
                int j = i + o;
                s += w[o + RADIUS] * (j >= 0 && j < n ? in[j] : 0.0f);
            }
            want[i] = s;
        }
    });
    Buffer<float> din(in), dout(n);
    double k = time_device([&] { stencil1d<<<(n + BLOCK1D - 1) / BLOCK1D, BLOCK1D>>>(din, dout, n); });
    report("stencil1d_r3_shared", close(dout.download().data(), want.data(), n, 1e-5, 1e-6), k, ref);
}

static void case_jacobi2d() {
    const int nx = 2048, ny = 2048, iters = 20;
    const size_t n = size_t(nx) * ny;
    auto init = random_floats(n, 21, 0.0f, 1.0f);
    std::vector<float> ha(n), hb(n);
    std::vector<float>* want = iters % 2 ? &ha : &hb;
    double ref = best_ms([&] {
#pragma omp parallel
        {
            const float* src = init.data();
            float* dst = ha.data();
            for (int it = 0; it < iters; ++it) {
#pragma omp for schedule(static)
                for (int y = 0; y < ny; ++y) {
                    float* d = dst + size_t(y) * nx;
                    const float* s = src + size_t(y) * nx;
                    if (y == 0 || y == ny - 1) {
                        for (int x = 0; x < nx; ++x) d[x] = s[x];
                        continue;
                    }
                    d[0] = s[0];
                    d[nx - 1] = s[nx - 1];
#pragma omp simd
                    for (int x = 1; x < nx - 1; ++x) d[x] = 0.25f * (s[x - 1] + s[x + 1] + s[x - nx] + s[x + nx]);
                }
                src = dst;
                dst = dst == ha.data() ? hb.data() : ha.data();
            }
        }
    });
    Buffer<float> d0(init), d1(n), d2(n);
    float* result = nullptr;
    dim3 block(32, 8), grid((nx + 31) / 32, (ny + 7) / 8);
    double k = time_device([&] {
        const float* src = d0;
        float* dst = d1;
        for (int it = 0; it < iters; ++it) {
            jacobi2d<<<grid, block>>>(src, dst, nx, ny);
            result = dst;
            src = dst;
            dst = dst == d1.p ? d2.p : d1.p;
        }
    });
    std::vector<float> got(n);
    STRESS_CHECK(cudaMemcpy(got.data(), result, n * sizeof(float), cudaMemcpyDeviceToHost));
    report("jacobi2d_20iters", close(got.data(), want->data(), n, 1e-5, 1e-6), k, ref);
}

static void case_stencil3d() {
    const int nx = 384, ny = 384, nz = 256;
    const size_t n = size_t(nx) * ny * nz, stride = size_t(nx) * ny;
    const float c0 = 0.4f, c1 = 0.1f;
    auto in = random_floats(n, 31);
    std::vector<float> want(n);
    double ref = best_ms([&] {
#pragma omp parallel for collapse(2)
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y) {
                size_t row = z * stride + size_t(y) * nx;
                if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                    for (int x = 0; x < nx; ++x) want[row + x] = in[row + x];
                    continue;
                }
                want[row] = in[row];
                want[row + nx - 1] = in[row + nx - 1];
#pragma omp simd
                for (int x = 1; x < nx - 1; ++x) {
                    size_t i = row + x;
                    want[i] = c0 * in[i] + c1 * (in[i - stride] + in[i + stride] + in[i - 1] + in[i + 1] +
                                                 in[i - nx] + in[i + nx]);
                }
            }
    });
    Buffer<float> din(in), dout(n);
    dim3 block(BX3, BY3), grid((nx + BX3 - 1) / BX3, (ny + BY3 - 1) / BY3);
    double k = time_device([&] { stencil3d<<<grid, block>>>(din, dout, nx, ny, nz, c0, c1); });
    report("stencil3d_7pt", close(dout.download().data(), want.data(), n, 1e-5, 1e-6), k, ref);
}

static void case_conv2d() {
    const int w = 4096, h = 4000;
    const size_t n = size_t(w) * h;
    auto in = random_floats(n, 41);
    auto filt = random_floats(FW * FW, 42, -0.2f, 0.2f);
    STRESS_CHECK(cudaMemcpyToSymbol(c_filter, filt.data(), FW * FW * sizeof(float)));
    std::vector<float> want(n);
    double ref = best_ms([&] {
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            float* o = &want[size_t(y) * w];
            for (int x = 0; x < w; ++x) o[x] = 0.0f;
            for (int ky = 0; ky < FW; ++ky) {
                int gy = y + ky - FR;
                if (gy < 0 || gy >= h) continue;
                const float* r = &in[size_t(gy) * w];
                for (int kx = 0; kx < FW; ++kx) {
                    float f = filt[ky * FW + kx];
                    int lo = std::max(0, FR - kx), hi = std::min(w, w + FR - kx);
#pragma omp simd
                    for (int x = lo; x < hi; ++x) o[x] += f * r[x + kx - FR];
                }
            }
        }
    });
    Buffer<float> din(in), dout(n);
    dim3 block(CTILE, CTILE), grid((w + CTILE - 1) / CTILE, (h + CTILE - 1) / CTILE);
    double k = time_device([&] { conv2d<<<grid, block>>>(din, dout, w, h); });
    report("conv2d_5x5_const", close(dout.download().data(), want.data(), n, 1e-4, 1e-5), k, ref);
}

static void case_gauss() {
    const int w = 4096, h = 4096;
    const size_t n = size_t(w) * h;
    auto in = random_floats(n, 51, 0.0f, 1.0f);
    float g[GW], sum = 0.0f;
    const float sigma = 3.0f;
    for (int k = 0; k < GW; ++k) sum += g[k] = std::exp(-float((k - GR) * (k - GR)) / (2 * sigma * sigma));
    for (int k = 0; k < GW; ++k) g[k] /= sum;
    STRESS_CHECK(cudaMemcpyToSymbol(c_gauss, g, sizeof(g)));
    std::vector<float> tmp(n), want(n);
    double ref = best_ms([&] {
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            const float* r = &in[size_t(y) * w];
            float* o = &tmp[size_t(y) * w];
#pragma omp simd
            for (int x = 0; x < w; ++x) {
                float s = 0.0f;
                for (int k = 0; k < GW; ++k) s += g[k] * r[std::min(std::max(x + k - GR, 0), w - 1)];
                o[x] = s;
            }
        }
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            float* o = &want[size_t(y) * w];
            for (int x = 0; x < w; ++x) o[x] = 0.0f;
            for (int k = 0; k < GW; ++k) {
                const float* r = &tmp[size_t(std::min(std::max(y + k - GR, 0), h - 1)) * w];
                float f = g[k];
#pragma omp simd
                for (int x = 0; x < w; ++x) o[x] += f * r[x];
            }
        }
    });
    Buffer<float> din(in), dtmp(n), dout(n);
    dim3 block(GBX, GBY), grid((w + GBX - 1) / GBX, (h + GBY - 1) / GBY);
    double k = time_device([&] {
        gauss_rows<<<grid, block>>>(din, dtmp, w, h);
        gauss_cols<<<grid, block>>>(dtmp, dout, w, h);
    });
    report("gauss_separable_r8", close(dout.download().data(), want.data(), n, 1e-5, 1e-6), k, ref);
}

static void case_sobel() {
    const int w = 8192, h = 6144;
    const size_t n = size_t(w) * h;
    std::vector<unsigned char> in(n);
#pragma omp parallel for
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint32_t r = hash32(uint32_t(y) * 2654435761u + x * 40503u + 61);
            in[size_t(y) * w + x] = (unsigned char)(((x / 3 + y / 5) & 255) / 2 + (r & 63));
        }
    std::vector<unsigned char> want(n);
    double ref = best_ms([&] {
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            unsigned char* o = &want[size_t(y) * w];
            if (y == 0 || y == h - 1) {
                for (int x = 0; x < w; ++x) o[x] = 0;
                continue;
            }
            const unsigned char *a = &in[size_t(y - 1) * w], *b = &in[size_t(y) * w], *c = &in[size_t(y + 1) * w];
            o[0] = o[w - 1] = 0;
#pragma omp simd
            for (int x = 1; x < w - 1; ++x) {
                int gx = (a[x + 1] + 2 * b[x + 1] + c[x + 1]) - (a[x - 1] + 2 * b[x - 1] + c[x - 1]);
                int gy = (c[x - 1] + 2 * c[x] + c[x + 1]) - (a[x - 1] + 2 * a[x] + a[x + 1]);
                float mag = std::sqrt(float(gx * gx + gy * gy));
                o[x] = (unsigned char)std::min(255, int(mag));
            }
        }
    });
    Buffer<unsigned char> din(in), dout(n);
    dim3 block(SBX, SBY), grid((w + SBX - 1) / SBX, (h + SBY - 1) / SBY);
    double k = time_device([&] { sobel<<<grid, block>>>(din, dout, w, h); });
    report("sobel_u8", equal(dout.download().data(), want.data(), n), k, ref);
}

int main() {
    case_stencil1d();
    case_jacobi2d();
    case_stencil3d();
    case_conv2d();
    case_gauss();
    case_sobel();
    return finish();
}
