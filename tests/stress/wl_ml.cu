// Machine-learning layers: row-wise softmax (rows of 1024 and 4096, one block
// per row, warp shuffles plus shared memory across warps), layer
// normalization, a 1D convolution layer with shared-memory tiling, 2x2 max
// pooling (NCHW), bias + ReLU, an LSTM cell elementwise kernel and a
// cross-entropy loss accumulated with atomicAdd.
#include "stress.cuh"

// ---------------------------------------------------------------------------
// Block reductions.

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Every thread of the block gets the result. `scratch` holds 32 floats.
template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();  // scratch may still be read by a previous reduction
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < BLOCK / 32 ? scratch[lane] : 0.0f;
    return warp_sum(v);
}

template <int BLOCK>
__device__ __forceinline__ float block_max(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < BLOCK / 32 ? scratch[lane] : -INFINITY;
    return warp_max(v);
}

// ---------------------------------------------------------------------------
// Softmax: one block per row, the row kept in registers as float4.

template <int COLS, int BLOCK>
__global__ void __launch_bounds__(BLOCK) softmax_rows(const float* __restrict__ in, float* __restrict__ out) {
    constexpr int VEC = COLS / (4 * BLOCK);
    __shared__ float scratch[32];
    const float4* row = reinterpret_cast<const float4*>(in + (size_t)blockIdx.x * COLS);
    float4* orow = reinterpret_cast<float4*>(out + (size_t)blockIdx.x * COLS);
    float4 v[VEC];
    float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
        v[k] = row[k * BLOCK + threadIdx.x];
        m = fmaxf(m, fmaxf(fmaxf(v[k].x, v[k].y), fmaxf(v[k].z, v[k].w)));
    }
    m = block_max<BLOCK>(m, scratch);
    float s = 0;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
        v[k].x = __expf(v[k].x - m);
        v[k].y = __expf(v[k].y - m);
        v[k].z = __expf(v[k].z - m);
        v[k].w = __expf(v[k].w - m);
        s += (v[k].x + v[k].y) + (v[k].z + v[k].w);
    }
    s = block_sum<BLOCK>(s, scratch);
    float inv = 1.0f / s;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
        float4 o = make_float4(v[k].x * inv, v[k].y * inv, v[k].z * inv, v[k].w * inv);
        orow[k * BLOCK + threadIdx.x] = o;
    }
}

// ---------------------------------------------------------------------------
// Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta.

template <int COLS, int BLOCK>
__global__ void __launch_bounds__(BLOCK) layer_norm(const float* __restrict__ in, float* __restrict__ out,
                                                    const float* __restrict__ gamma, const float* __restrict__ beta,
                                                    float eps) {
    constexpr int VEC = COLS / (4 * BLOCK);
    __shared__ float scratch[32];
    const float4* row = reinterpret_cast<const float4*>(in + (size_t)blockIdx.x * COLS);
    float4* orow = reinterpret_cast<float4*>(out + (size_t)blockIdx.x * COLS);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    float4 v[VEC];
    float s = 0;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
        v[k] = row[k * BLOCK + threadIdx.x];
        s += (v[k].x + v[k].y) + (v[k].z + v[k].w);
    }
    float mean = block_sum<BLOCK>(s, scratch) * (1.0f / COLS);
    float q = 0;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
        v[k].x -= mean;
        v[k].y -= mean;
        v[k].z -= mean;
        v[k].w -= mean;
        q += (v[k].x * v[k].x + v[k].y * v[k].y) + (v[k].z * v[k].z + v[k].w * v[k].w);
    }
    float var = block_sum<BLOCK>(q, scratch) * (1.0f / COLS);
    float rstd = rsqrtf(var + eps);
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
        int idx = k * BLOCK + threadIdx.x;
        float4 g = g4[idx], b = b4[idx], o;
        o.x = v[k].x * rstd * g.x + b.x;
        o.y = v[k].y * rstd * g.y + b.y;
        o.z = v[k].z * rstd * g.z + b.z;
        o.w = v[k].w * rstd * g.w + b.w;
        orow[idx] = o;
    }
}

// ---------------------------------------------------------------------------
// Conv1d, "same" padding: out[b][co][l] = bias[co] + sum_ci sum_k w[co][ci][k] * in[b][ci][l + k - K/2].
// Block: TL output positions x CO_TILE output channels; input and weight
// tiles staged in shared memory, CI_TILE input channels at a time.

template <int K, int TL, int CO_TILE, int CI_TILE>
__global__ void __launch_bounds__(TL) conv1d(const float* __restrict__ in, const float* __restrict__ w,
                                             const float* __restrict__ bias, float* __restrict__ out, int cin, int cout,
                                             int L) {
    constexpr int PAD = K / 2;
    __shared__ float s_in[CI_TILE][TL + K - 1];
    __shared__ float s_w[CO_TILE][CI_TILE][K];
    const int b = blockIdx.z, co0 = blockIdx.y * CO_TILE, l0 = blockIdx.x * TL;
    const int t = threadIdx.x;
    const float* inb = in + (size_t)b * cin * L;
    float acc[CO_TILE];
#pragma unroll
    for (int c = 0; c < CO_TILE; ++c) acc[c] = 0.0f;

    for (int ci0 = 0; ci0 < cin; ci0 += CI_TILE) {
        for (int i = t; i < CI_TILE * (TL + K - 1); i += TL) {
            int ci = i / (TL + K - 1), x = i % (TL + K - 1);
            int l = l0 + x - PAD;
            s_in[ci][x] = (l >= 0 && l < L && ci0 + ci < cin) ? inb[(size_t)(ci0 + ci) * L + l] : 0.0f;
        }
        for (int i = t; i < CO_TILE * CI_TILE * K; i += TL) {
            int co = i / (CI_TILE * K), r = i % (CI_TILE * K), ci = r / K, k = r % K;
            s_w[co][ci][k] = (co0 + co < cout && ci0 + ci < cin) ? w[((size_t)(co0 + co) * cin + ci0 + ci) * K + k] : 0.0f;
        }
        __syncthreads();
#pragma unroll 4
        for (int ci = 0; ci < CI_TILE; ++ci) {
            float x[K];
#pragma unroll
            for (int k = 0; k < K; ++k) x[k] = s_in[ci][t + k];
#pragma unroll
            for (int c = 0; c < CO_TILE; ++c)
#pragma unroll
                for (int k = 0; k < K; ++k) acc[c] += s_w[c][ci][k] * x[k];
        }
        __syncthreads();
    }
    int l = l0 + t;
    if (l < L)
#pragma unroll
        for (int c = 0; c < CO_TILE; ++c)
            if (co0 + c < cout) out[((size_t)b * cout + co0 + c) * L + l] = acc[c] + bias[co0 + c];
}

// ---------------------------------------------------------------------------
// Max pooling 2x2, stride 2, NCHW.

__global__ void maxpool2x2(const float* __restrict__ in, float* __restrict__ out, int nc, int H, int W) {
    int Ho = H / 2, Wo = W / 2;
    size_t total = (size_t)nc * Ho * Wo;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        int x = i % Wo;
        int y = (i / Wo) % Ho;
        size_t p = i / ((size_t)Wo * Ho);
        const float* src = in + (p * H + 2 * y) * W + 2 * x;
        float2 r0 = *reinterpret_cast<const float2*>(src);
        float2 r1 = *reinterpret_cast<const float2*>(src + W);
        out[i] = fmaxf(fmaxf(r0.x, r0.y), fmaxf(r1.x, r1.y));
    }
}

// ---------------------------------------------------------------------------
// Bias + ReLU, NCHW with per-channel bias, float4.

__global__ void bias_relu(const float4* __restrict__ in, float4* __restrict__ out, const float* __restrict__ bias, int C,
                          int hw4, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float b = bias[(i / hw4) % C];
        float4 v = in[i];
        v.x = fmaxf(v.x + b, 0.0f);
        v.y = fmaxf(v.y + b, 0.0f);
        v.z = fmaxf(v.z + b, 0.0f);
        v.w = fmaxf(v.w + b, 0.0f);
        out[i] = v;
    }
}

// ---------------------------------------------------------------------------
// LSTM cell, elementwise part: gates[b][4H] = (i, f, g, o) pre-activations
// (from the GEMMs), plus a per-gate bias.

__device__ __forceinline__ float sigmoidf_dev(float x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void lstm_cell(const float* __restrict__ gates, const float* __restrict__ bias, const float* __restrict__ c_prev,
                          float* __restrict__ c_out, float* __restrict__ h_out, int B, int H) {
    size_t total = (size_t)B * H;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int b = idx / H, j = idx % H;
        const float* g = gates + (size_t)b * 4 * H;
        float ig = sigmoidf_dev(g[j] + bias[j]);
        float fg = sigmoidf_dev(g[H + j] + bias[H + j]);
        float gg = tanhf(g[2 * H + j] + bias[2 * H + j]);
        float og = sigmoidf_dev(g[3 * H + j] + bias[3 * H + j]);
        float c = fg * c_prev[idx] + ig * gg;
        c_out[idx] = c;
        h_out[idx] = og * tanhf(c);
    }
}

// ---------------------------------------------------------------------------
// Cross-entropy: loss = mean_r (logsumexp(x_r) - x_r[target_r]); one block per
// row, one atomicAdd per row.

template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) cross_entropy(const float* __restrict__ logits, const int* __restrict__ target,
                                                       float* __restrict__ loss, int C, float inv_rows) {
    __shared__ float scratch[32];
    const float* row = logits + (size_t)blockIdx.x * C;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < C; j += BLOCK) m = fmaxf(m, row[j]);
    m = block_max<BLOCK>(m, scratch);
    float s = 0;
    for (int j = threadIdx.x; j < C; j += BLOCK) s += __expf(row[j] - m);
    s = block_sum<BLOCK>(s, scratch);
    if (threadIdx.x == 0) {
        float nll = m + __logf(s) - row[target[blockIdx.x]];
        atomicAdd(loss, nll * inv_rows);
    }
}

// ---------------------------------------------------------------------------
// Host references.

static void softmax_ref(const float* in, float* out, int rows, int cols) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* x = in + (size_t)r * cols;
        float* y = out + (size_t)r * cols;
        float m = -INFINITY;
#pragma omp simd reduction(max : m)
        for (int j = 0; j < cols; ++j) m = std::max(m, x[j]);
        float s = 0;
#pragma omp simd reduction(+ : s)
        for (int j = 0; j < cols; ++j) {
            y[j] = std::exp(x[j] - m);
            s += y[j];
        }
        float inv = 1.0f / s;
#pragma omp simd
        for (int j = 0; j < cols; ++j) y[j] *= inv;
    }
}

template <int COLS, int BLOCK>
static void softmax_case(const char* name, int rows, uint32_t seed) {
    using namespace stress;
    size_t n = (size_t)rows * COLS;
    auto x = random_floats(n, seed, -8.0f, 8.0f);
    std::vector<float> want(n);
    double ref = best_ms([&] { softmax_ref(x.data(), want.data(), rows, COLS); });
    Buffer<float> din(x), dout(n);
    double k = time_device([&] { softmax_rows<COLS, BLOCK><<<rows, BLOCK>>>(din, dout); });
    auto got = dout.download();
    report(name, close(got.data(), want.data(), n, 1e-4, 1e-9, "softmax"), k, ref);
}

// ---------------------------------------------------------------------------

int main() {
    using namespace stress;

    softmax_case<1024, 256>("softmax_rows1024", 16384, 101);
    softmax_case<4096, 256>("softmax_rows4096", 4096, 102);

    // Layer normalization.
    {
        constexpr int COLS = 1024, BLOCK = 256;
        const int rows = 16384;
        const float eps = 1e-5f;
        size_t n = (size_t)rows * COLS;
        auto x = random_floats(n, 201, -3.0f, 5.0f);
        auto gamma = random_floats(COLS, 202, 0.5f, 1.5f);
        auto beta = random_floats(COLS, 203, -0.5f, 0.5f);
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; ++r) {
                const float* xr = x.data() + (size_t)r * COLS;
                float* yr = want.data() + (size_t)r * COLS;
                float s = 0;
#pragma omp simd reduction(+ : s)
                for (int j = 0; j < COLS; ++j) s += xr[j];
                float mean = s / COLS;
                float q = 0;
#pragma omp simd reduction(+ : q)
                for (int j = 0; j < COLS; ++j) q += (xr[j] - mean) * (xr[j] - mean);
                float rstd = 1.0f / std::sqrt(q / COLS + eps);
#pragma omp simd
                for (int j = 0; j < COLS; ++j) yr[j] = (xr[j] - mean) * rstd * gamma[j] + beta[j];
            }
        });
        Buffer<float> din(x), dout(n), dg(gamma), db(beta);
        double k = time_device([&] { layer_norm<COLS, BLOCK><<<rows, BLOCK>>>(din, dout, dg, db, eps); });
        auto got = dout.download();
        report("layernorm_rows1024", close(got.data(), want.data(), n, 1e-4, 1e-4, "layernorm"), k, ref);
    }

    // Conv1d.
    {
        constexpr int K = 5, TL = 128, CO_TILE = 16, CI_TILE = 16;
        const int B = 8, cin = 64, cout = 128, L = 4096;
        auto x = random_floats((size_t)B * cin * L, 301);
        auto w = random_floats((size_t)cout * cin * K, 302, -0.2f, 0.2f);
        auto bias = random_floats(cout, 303, -0.1f, 0.1f);
        std::vector<float> want((size_t)B * cout * L);
        double ref = best_ms([&] {
#pragma omp parallel for collapse(2) schedule(static)
            for (int b = 0; b < B; ++b)
                for (int co = 0; co < cout; ++co) {
                    float* o = want.data() + ((size_t)b * cout + co) * L;
                    for (int l = 0; l < L; ++l) o[l] = bias[co];
                    for (int ci = 0; ci < cin; ++ci) {
                        const float* xi = x.data() + ((size_t)b * cin + ci) * L;
                        const float* wk = w.data() + ((size_t)co * cin + ci) * K;
                        for (int k = 0; k < K; ++k) {
                            int lo = std::max(0, K / 2 - k), hi = std::min(L, L + K / 2 - k);
                            float wv = wk[k];
#pragma omp simd
                            for (int l = lo; l < hi; ++l) o[l] += wv * xi[l + k - K / 2];
                        }
                    }
                }
        });
        Buffer<float> dx(x), dw(w), db(bias), dout(want.size());
        dim3 grid((L + TL - 1) / TL, (cout + CO_TILE - 1) / CO_TILE, B);
        double k = time_device([&] {
            conv1d<K, TL, CO_TILE, CI_TILE><<<grid, TL>>>(dx, dw, db, dout, cin, cout, L);
        });
        auto got = dout.download();
        report("conv1d_k5", close(got.data(), want.data(), want.size(), 1e-4, 2e-5, "conv1d"), k, ref);
    }

    // Max pooling 2x2.
    {
        const int N = 8, C = 64, H = 256, W = 256, Ho = H / 2, Wo = W / 2;
        size_t nin = (size_t)N * C * H * W, nout = (size_t)N * C * Ho * Wo;
        auto x = random_floats(nin, 401, -10.0f, 10.0f);
        std::vector<float> want(nout);
        double ref = best_ms([&] {
#pragma omp parallel for collapse(2) schedule(static)
            for (int p = 0; p < N * C; ++p)
                for (int y = 0; y < Ho; ++y) {
                    const float* r0 = x.data() + ((size_t)p * H + 2 * y) * W;
                    const float* r1 = r0 + W;
                    float* o = want.data() + ((size_t)p * Ho + y) * Wo;
#pragma omp simd
                    for (int xo = 0; xo < Wo; ++xo)
                        o[xo] = std::max(std::max(r0[2 * xo], r0[2 * xo + 1]), std::max(r1[2 * xo], r1[2 * xo + 1]));
                }
        });
        Buffer<float> din(x), dout(nout);
        double k = time_device([&] { maxpool2x2<<<(nout + 255) / 256, 256>>>(din, dout, N * C, H, W); });
        auto got = dout.download();
        report("maxpool2x2_nchw", equal(got.data(), want.data(), nout, "maxpool"), k, ref);
    }

    // Bias + ReLU.
    {
        const int N = 16, C = 64, HW = 128 * 128;
        size_t n = (size_t)N * C * HW;
        auto x = random_floats(n, 501, -1.0f, 1.0f);
        auto bias = random_floats(C, 502, -0.5f, 0.5f);
        std::vector<float> want(n);
        double ref = best_ms([&] {
#pragma omp parallel for schedule(static)
            for (int p = 0; p < N * C; ++p) {
                float b = bias[p % C];
                const float* xi = x.data() + (size_t)p * HW;
                float* o = want.data() + (size_t)p * HW;
#pragma omp simd
                for (int j = 0; j < HW; ++j) o[j] = std::max(xi[j] + b, 0.0f);
            }
        });
        Buffer<float> din(x), dout(n), db(bias);
        size_t n4 = n / 4;
        double k = time_device([&] {
            bias_relu<<<(n4 + 255) / 256, 256>>>(reinterpret_cast<const float4*>(din.p), reinterpret_cast<float4*>(dout.p),
                                                  db, C, HW / 4, n4);
        });
        auto got = dout.download();
        report("bias_relu_nchw", equal(got.data(), want.data(), n, "relu"), k, ref);
    }

    // LSTM cell.
    {
        const int B = 1024, H = 4096;
        size_t n = (size_t)B * H;
        auto gates = random_floats(4 * n, 601, -4.0f, 4.0f);
        auto bias = random_floats(4 * H, 602, -0.5f, 0.5f);
        auto cprev = random_floats(n, 603, -2.0f, 2.0f);
        std::vector<float> wc(n), wh(n);
        auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
        double ref = best_ms([&] {
#pragma omp parallel for schedule(static)
            for (int b = 0; b < B; ++b) {
                const float* g = gates.data() + (size_t)b * 4 * H;
#pragma omp simd
                for (int j = 0; j < H; ++j) {
                    size_t idx = (size_t)b * H + j;
                    float ig = sig(g[j] + bias[j]);
                    float fg = sig(g[H + j] + bias[H + j]);
                    float gg = std::tanh(g[2 * H + j] + bias[2 * H + j]);
                    float og = sig(g[3 * H + j] + bias[3 * H + j]);
                    float c = fg * cprev[idx] + ig * gg;
                    wc[idx] = c;
                    wh[idx] = og * std::tanh(c);
                }
            }
        });
        Buffer<float> dg(gates), db(bias), dcp(cprev), dc(n), dh(n);
        double k = time_device([&] { lstm_cell<<<(n + 255) / 256, 256>>>(dg, db, dcp, dc, dh, B, H); });
        auto gc = dc.download();
        auto gh = dh.download();
        bool ok = close(gc.data(), wc.data(), n, 1e-5, 2e-6, "c") && close(gh.data(), wh.data(), n, 1e-5, 2e-6, "h");
        report("lstm_cell", ok, k, ref);
    }

    // Cross-entropy loss.
    {
        constexpr int BLOCK = 256;
        const int rows = 16384, C = 1000;
        auto x = random_floats((size_t)rows * C, 701, -6.0f, 6.0f);
        auto tu = random_u32(rows, 702);
        std::vector<int> target(rows);
        for (int r = 0; r < rows; ++r) target[r] = tu[r] % C;
        double want = 0;
        double ref = best_ms([&] {
            double total = 0;
#pragma omp parallel for reduction(+ : total) schedule(static)
            for (int r = 0; r < rows; ++r) {
                const float* xr = x.data() + (size_t)r * C;
                float m = -INFINITY;
#pragma omp simd reduction(max : m)
                for (int j = 0; j < C; ++j) m = std::max(m, xr[j]);
                float s = 0;
#pragma omp simd reduction(+ : s)
                for (int j = 0; j < C; ++j) s += std::exp(xr[j] - m);
                total += m + std::log(s) - xr[target[r]];
            }
            want = total / rows;
        });
        Buffer<float> dx(x), dloss(1);
        Buffer<int> dt(target);
        double k = time_device([&] {
            STRESS_CHECK(cudaMemset(dloss, 0, sizeof(float)));
            cross_entropy<BLOCK><<<rows, BLOCK>>>(dx, dt, dloss, C, 1.0f / rows);
        });
        auto got = dloss.download();
        char note[64];
        std::snprintf(note, sizeof note, "loss=%.6f want=%.6f", got[0], want);
        report("cross_entropy_atomic", std::fabs(got[0] - want) <= 1e-4 * std::fabs(want), k, ref, note);
    }

    return finish();
}
