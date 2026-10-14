// Per-thread arrays: indexed by a runtime-sized loop (uniform index), by a
// per-thread index (gather/scatter), live across __syncthreads (the frame of
// the variants that suspend), of doubles, and struct copies (memcpy into a
// local). Each has a host reference.
#include <cstring>
#include <vector>

#include "check.cuh"

struct Particle {
    float pos[3];
    float vel[3];
    float mass;
};

__global__ void arrays(const float* in, float* out, double* outd, int n, int reps) {
    __shared__ float sh[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    float a[16];
    double d[8];
    for (int j = 0; j < 16; ++j) a[j] = 0.0f;
    for (int j = 0; j < reps; ++j) a[j % 16] += in[(i + j) % n] * float(j + 1);  // runtime trip count
    for (int j = 0; j < 8; ++j) d[j] = in[i % n] + j;
    float s = a[(t * 7) % 16] + a[t % 5];  // per-thread index
    sh[t] = s;
    __syncthreads();                       // a[] lives across the barrier
    float s2 = sh[(t + 1) % blockDim.x] + a[(t + 3) % 16];
    a[t % 16] = s2;                        // per-thread store
    double sd = 0;
    for (int j = 0; j < 8; ++j) sd += d[(j + t) % 8] * (j + 1);
    out[i] = s2 + a[(t + 1) % 16];
    outd[i] = sd;
}

__global__ void structs(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle me = p[i];  // struct copy into a local
    float acc[3] = {0, 0, 0};
    for (int j = 0; j < n; j += 7) {
        Particle o = p[j];
        for (int k = 0; k < 3; ++k) acc[k] += (o.pos[k] - me.pos[k]) * o.mass;
    }
    for (int k = 0; k < 3; ++k) me.vel[k] += acc[k];
    p[i] = me;  // and back
}

// A per-thread struct of mixed field sizes, written as fields and read back
// as bytes (each lane gets a real copy), alive across a barrier.
struct Mixed {
    int a;
    char tag[3];
    double d;
    short s;
};
__global__ void mixed_bytes(const int* in, unsigned* out) {
    __shared__ int sh[128];
    const int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    Mixed m;
    m.a = in[i];
    for (int k = 0; k < 3; ++k) m.tag[k] = char(in[i] + k);
    m.d = in[i] * 0.5;
    m.s = short(in[i] * 3);
    sh[t] = in[i];
    __syncthreads();  // m lives across the barrier
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&m);
    unsigned h = unsigned(sh[(t + 1) % blockDim.x]);
    for (int k = 0; k < 4; ++k) h = h * 31 + b[k];        // a
    for (int k = 4; k < 7; ++k) h = h * 31 + b[k];        // tag
    for (int k = 8; k < 16; ++k) h = h * 31 + b[k];       // d
    for (int k = 16; k < 18; ++k) h = h * 31 + b[k];      // s
    out[i] = h;
}

int main() {
    Checker c("local_arrays");
    {
        const int nb = 4, nt = 128, nm = nb * nt;
        std::vector<int> in(nm);
        for (int i = 0; i < nm; ++i) in[i] = (i * 7919) % 100003 - 50000;
        int* din;
        unsigned* dout;
        CHECK_CUDA(cudaMalloc(&din, nm * 4));
        CHECK_CUDA(cudaMalloc(&dout, nm * 4));
        CHECK_CUDA(cudaMemcpy(din, in.data(), nm * 4, cudaMemcpyHostToDevice));
        mixed_bytes<<<nb, nt>>>(din, dout);
        std::vector<unsigned> out(nm);
        CHECK_CUDA(cudaMemcpy(out.data(), dout, nm * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < nm; ++i) {
            Mixed m;
            std::memset(&m, 0, sizeof(m));
            m.a = in[i];
            for (int k = 0; k < 3; ++k) m.tag[k] = char(in[i] + k);
            m.d = in[i] * 0.5;
            m.s = short(in[i] * 3);
            const int t = i % nt;
            const unsigned char* b = reinterpret_cast<const unsigned char*>(&m);
            unsigned h = unsigned(in[(i / nt) * nt + (t + 1) % nt]);
            for (int k = 0; k < 7; ++k) h = h * 31 + b[k];
            for (int k = 8; k < 18; ++k) h = h * 31 + b[k];
            c.expect(out[i] == h, "per-thread struct read as bytes across a barrier", i, out[i], h);
        }
        CHECK_CUDA(cudaFree(din));
        CHECK_CUDA(cudaFree(dout));
    }
    const int threads = 256, blocks = 16, n = threads * blocks, reps = 37;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = float(i % 97) * 0.25f;
    float *din, *dout;
    double* doutd;
    CHECK_CUDA(cudaMalloc(&din, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&doutd, n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(din, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    arrays<<<blocks, threads>>>(din, dout, doutd, n, reps);
    CHECK_CUDA(cudaGetLastError());
    std::vector<float> out(n);
    std::vector<double> outd(n);
    CHECK_CUDA(cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(outd.data(), doutd, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Host model: every thread's array, then the block exchange.
    std::vector<float> s(n), arr(size_t(n) * 16);
    for (int i = 0; i < n; ++i) {
        float* a = &arr[size_t(i) * 16];
        for (int j = 0; j < 16; ++j) a[j] = 0.0f;
        for (int j = 0; j < reps; ++j) a[j % 16] += in[(i + j) % n] * float(j + 1);
        int t = i % threads;
        s[i] = a[(t * 7) % 16] + a[t % 5];
    }
    for (int i = 0; i < n; ++i) {
        int t = i % threads, b0 = i - t;
        float* a = &arr[size_t(i) * 16];
        float s2 = s[b0 + (t + 1) % threads] + a[(t + 3) % 16];
        a[t % 16] = s2;
        float want = s2 + a[(t + 1) % 16];
        c.expect(out[i] == want, "float array", i, out[i], want);
        double sd = 0;
        for (int j = 0; j < 8; ++j) sd += (in[i % n] + (j + t) % 8) * (j + 1);
        c.expect(outd[i] == sd, "double array", i, outd[i], sd);
    }

    const int np = 1000;
    std::vector<Particle> ps(np);
    for (int i = 0; i < np; ++i)
        ps[i] = {{float(i % 13), float(i % 7), float(i % 5)}, {1, 2, 3}, 0.5f + float(i % 3)};
    Particle* dp;
    CHECK_CUDA(cudaMalloc(&dp, np * sizeof(Particle)));
    CHECK_CUDA(cudaMemcpy(dp, ps.data(), np * sizeof(Particle), cudaMemcpyHostToDevice));
    structs<<<(np + 127) / 128, 128>>>(dp, np);
    CHECK_CUDA(cudaGetLastError());
    std::vector<Particle> got(np);
    CHECK_CUDA(cudaMemcpy(got.data(), dp, np * sizeof(Particle), cudaMemcpyDeviceToHost));
    for (int i = 0; i < np; ++i) {
        float acc[3] = {0, 0, 0};
        for (int j = 0; j < np; j += 7)
            for (int k = 0; k < 3; ++k) acc[k] += (ps[j].pos[k] - ps[i].pos[k]) * ps[j].mass;
        for (int k = 0; k < 3; ++k) {
            float want = ps[i].vel[k] + acc[k];
            c.expect(close_rel(got[i].vel[k], want, 1e-5, 1e-4), "struct copy", i, got[i].vel[k], want);
            c.expect(got[i].pos[k] == ps[i].pos[k], "struct copy keeps pos", i, got[i].pos[k], ps[i].pos[k]);
        }
    }
    cudaFree(din);
    cudaFree(dout);
    cudaFree(doutd);
    cudaFree(dp);
    return c.finish();
}
