// Sparse matrix-vector product y = A * x with A in CSR format.
//
// Exercises: irregular memory access (x[col_idx[k]] is a true gather),
// variable per-lane loop trip counts (rows have 0..64 non-zeros, so lanes
// leave the loop at different times), and a warp-per-row variant that uses
// __shfl_down_sync with a warp-strided inner loop.
#include "common.cuh"

constexpr int SPMV_THREADS = 256;
constexpr int MAX_ROW_NNZ = 64;

__global__ void spmv_csr_scalar(const int* row_ptr, const int* col_idx, const float* vals,
                                const float* x, float* y, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float sum = 0.0f;
    for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k) sum += vals[k] * x[col_idx[k]];
    y[r] = sum;
}

__global__ void spmv_csr_warp(const int* row_ptr, const int* col_idx, const float* vals,
                              const float* x, float* y, int rows) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // one warp per row
    int lane = threadIdx.x & 31;
    if (r >= rows) return;  // uniform across the warp

    float sum = 0.0f;
    for (int k = row_ptr[r] + lane; k < row_ptr[r + 1]; k += 32) sum += vals[k] * x[col_idx[k]];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) y[r] = sum;
}

int main() {
    const int rows = 20000, cols = 20000;

    Lcg rng(17);
    std::vector<int> row_ptr(rows + 1, 0), col_idx;
    std::vector<float> vals;
    for (int r = 0; r < rows; ++r) {
        int nnz = static_cast<int>(rng.next_u32() % (MAX_ROW_NNZ + 1));
        for (int k = 0; k < nnz; ++k) {
            col_idx.push_back(static_cast<int>(rng.next_u32() % cols));
            vals.push_back(rng.next_float(-1.0f, 1.0f));
        }
        row_ptr[r + 1] = static_cast<int>(col_idx.size());
    }
    std::vector<float> x(cols);
    for (auto& v : x) v = rng.next_float(-1.0f, 1.0f);

    std::vector<float> ref(rows);
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k) sum += double(vals[k]) * x[col_idx[k]];
        ref[r] = static_cast<float>(sum);
    }

    DeviceBuffer<int> d_row_ptr(row_ptr.size()), d_col_idx(col_idx.size());
    DeviceBuffer<float> d_vals(vals.size()), d_x(x.size()), d_y(rows);
    d_row_ptr.upload(row_ptr);
    d_col_idx.upload(col_idx);
    d_vals.upload(vals);
    d_x.upload(x);
    std::vector<float> got;
    bool ok = true;

    report_time("spmv_csr_scalar", time_ms([&] {
        spmv_csr_scalar<<<(rows + SPMV_THREADS - 1) / SPMV_THREADS, SPMV_THREADS>>>(
            d_row_ptr.get(), d_col_idx.get(), d_vals.get(), d_x.get(), d_y.get(), rows);
    }));
    d_y.download(got);
    ok &= check_close(ref.data(), got.data(), rows, 1e-4, 1e-5, "spmv_csr_scalar");

    d_y.zero();
    int warps_per_block = SPMV_THREADS / 32;
    report_time("spmv_csr_warp", time_ms([&] {
        spmv_csr_warp<<<(rows + warps_per_block - 1) / warps_per_block, SPMV_THREADS>>>(
            d_row_ptr.get(), d_col_idx.get(), d_vals.get(), d_x.get(), d_y.get(), rows);
    }));
    d_y.download(got);
    ok &= check_close(ref.data(), got.data(), rows, 1e-4, 1e-5, "spmv_csr_warp");

    return finish("spmv", ok);
}
