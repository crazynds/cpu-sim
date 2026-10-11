// Device math library against the host's libm: functions with an AVX-512
// instruction (sqrt, fabs, floor, fma, fmin...) and the transcendental ones
// lowered to libmvec (exp, log, pow, sin, erf, atan2...), float and double,
// with per-lane (varying) and per-warp (uniform) arguments, plus the CUDA fast
// forms (__expf, rsqrtf, __fdividef).
#include <vector>

#include "check.cuh"

constexpr int kF = 22, kD = 8;

__global__ void math_f(const float* x, const float* y, float* out, float u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = x[i], b = y[i];
    float* o = out + size_t(i) * kF;
    o[0] = sqrtf(a);
    o[1] = rsqrtf(a + 0.5f);
    o[2] = fabsf(b);
    o[3] = floorf(b);
    o[4] = fmaf(a, b, 1.0f);
    o[5] = fminf(a, b);
    o[6] = expf(b);
    o[7] = logf(a + 0.1f);
    o[8] = powf(a, 1.7f);
    o[9] = sinf(b);
    o[10] = cosf(b);
    o[11] = tanhf(b);
    o[12] = erff(b);
    o[13] = erfcf(b);
    o[14] = atan2f(b, a + 0.1f);
    o[15] = log2f(a + 1.0f);
    o[16] = exp2f(b);
    o[17] = cbrtf(b);
    o[18] = __expf(b);
    o[19] = __fdividef(b, a + 1.0f);
    o[20] = expf(u) + logf(u);  // uniform argument
    o[21] = exp(b);             // C++ overload: float version
}

__global__ void math_d(const double* x, const double* y, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double a = x[i], b = y[i];
    double* o = out + size_t(i) * kD;
    o[0] = sqrt(a);
    o[1] = exp(b);
    o[2] = log(a + 0.1);
    o[3] = pow(a, 1.7);
    o[4] = sin(b);
    o[5] = tanh(b);
    o[6] = erf(b);
    o[7] = atan2(b, a + 0.1);
}

int main() {
    Checker c("math");
    const int threads = 128, blocks = 8, n = threads * blocks;
    std::vector<float> x(n), y(n);
    std::vector<double> xd(n), yd(n);
    for (int i = 0; i < n; ++i) {
        x[i] = float(i) / n * 9.0f;           // [0, 9)
        y[i] = (float(i % 97) - 48.0f) / 9.0f;  // about [-5.3, 5.4]
        xd[i] = x[i];
        yd[i] = y[i];
    }
    const float u = 2.5f;
    float *dx, *dy, *dout;
    double *ddx, *ddy, *ddout;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, size_t(n) * kF * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ddx, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&ddy, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&ddout, size_t(n) * kD * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ddx, xd.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ddy, yd.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    math_f<<<blocks, threads>>>(dx, dy, dout, u);
    math_d<<<blocks, threads>>>(ddx, ddy, ddout);
    CHECK_CUDA(cudaGetLastError());
    std::vector<float> o(size_t(n) * kF);
    std::vector<double> od(size_t(n) * kD);
    CHECK_CUDA(cudaMemcpy(o.data(), dout, o.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(od.data(), ddout, od.size() * sizeof(double), cudaMemcpyDeviceToHost));

    static const char* fnames[kF] = {"sqrtf", "rsqrtf", "fabsf", "floorf", "fmaf", "fminf", "expf", "logf",
                                     "powf", "sinf", "cosf", "tanhf", "erff", "erfcf", "atan2f", "log2f",
                                     "exp2f", "cbrtf", "__expf", "__fdividef", "uniform exp+log", "exp(float)"};
    for (int i = 0; i < n; ++i) {
        double a = x[i], b = y[i];
        double want[kF] = {std::sqrt(a), 1 / std::sqrt(a + 0.5), std::fabs(b), std::floor(b), a * b + 1,
                           std::fmin(a, b), std::exp(b), std::log(a + 0.1), std::pow(a, 1.7), std::sin(b),
                           std::cos(b), std::tanh(b), std::erf(b), std::erfc(b), std::atan2(b, a + 0.1),
                           std::log2(a + 1), std::exp2(b), std::cbrt(b), std::exp(b), b / (a + 1),
                           std::exp(double(u)) + std::log(double(u)), std::exp(b)};
        // libmvec's float functions are within 4 ulp: ~5e-7 relative; values
        // near zero (sin, erf, atan2) get an absolute tolerance too.
        for (int k = 0; k < kF; ++k)
            c.expect(close_rel(o[size_t(i) * kF + k], want[k], 2e-6, 1e-6), fnames[k], i, o[size_t(i) * kF + k], want[k]);
        double wd[kD] = {std::sqrt(a), std::exp(b), std::log(a + 0.1), std::pow(a, 1.7),
                         std::sin(b), std::tanh(b), std::erf(b), std::atan2(b, a + 0.1)};
        static const char* dnames[kD] = {"sqrt", "exp", "log", "pow", "sin", "tanh", "erf", "atan2"};
        for (int k = 0; k < kD; ++k)
            c.expect(close_rel(od[size_t(i) * kD + k], wd[k], 1e-14, 1e-15), dnames[k], i, od[size_t(i) * kD + k], wd[k]);
    }
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dout);
    cudaFree(ddx);
    cudaFree(ddy);
    cudaFree(ddout);
    return c.finish();
}
