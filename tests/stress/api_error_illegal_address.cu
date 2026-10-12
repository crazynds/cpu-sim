// A kernel that writes far outside any allocation (address 0x10). On a GPU
// this is a sticky error: the synchronize reports cudaErrorIllegalAddress and
// the context is unusable afterwards, so this program has a single case and
// only checks the error codes.
#include "stress.cuh"

__global__ void write_through(int* p, int v) {
    if (threadIdx.x == 0) {
        // The pointer value comes from the host, so the compiler cannot see it is bad.
        p[0] = v;
    }
}

__global__ void good(int* p) { p[threadIdx.x] = threadIdx.x; }

int main() {
    using namespace stress;
    // A good kernel first, so the context exists and works.
    int* d;
    STRESS_CHECK(cudaMalloc(&d, 64 * sizeof(int)));
    good<<<1, 64>>>(d);
    STRESS_CHECK(cudaGetLastError());
    STRESS_CHECK(cudaDeviceSynchronize());

    double t0 = now_ms();
    write_through<<<1, 32>>>(reinterpret_cast<int*>(uintptr_t(0x10)), 42);
    cudaError_t launch = cudaGetLastError();  // the launch itself is valid
    cudaError_t sync_err = cudaDeviceSynchronize();
    double k = now_ms() - t0;
    std::printf("  launch=%s sync=%s\n", cudaGetErrorName(launch), cudaGetErrorName(sync_err));
    // Sticky: reported again by the next synchronize.
    cudaError_t again = cudaDeviceSynchronize();
    bool ok = launch == cudaSuccess && sync_err == cudaErrorIllegalAddress && again == cudaErrorIllegalAddress;
    report("kernel_illegal_address", ok, k, -1);
    return finish();
}
