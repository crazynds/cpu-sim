// All-pairs N-body simulation (gravity with softening), tiled through shared memory.
//
// Exercises: float4 loads/stores, a __host__ __device__ helper shared with the
// CPU reference, a shared-memory tile read with a uniform index (every lane
// reads tile[k]: a broadcast), a loop whose trip count is uniform per block,
// rsqrtf, and ping-pong buffers across several launches.
#include "common.cuh"

constexpr int BLOCK = 128;
constexpr float SOFTENING2 = 0.01f;

__host__ __device__ inline float inv_sqrt(float x) {
#ifdef __CUDA_ARCH__
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

__host__ __device__ inline float3 body_interaction(float4 bi, float4 bj, float3 acc) {
    float rx = bj.x - bi.x;
    float ry = bj.y - bi.y;
    float rz = bj.z - bi.z;
    float dist2 = rx * rx + ry * ry + rz * rz + SOFTENING2;
    float inv = inv_sqrt(dist2);
    float s = bj.w * (inv * inv * inv);
    acc.x += rx * s;
    acc.y += ry * s;
    acc.z += rz * s;
    return acc;
}

__host__ __device__ inline void integrate(float4 pos, float4 vel, float3 acc, float dt,
                                          float4& pos_out, float4& vel_out) {
    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;
    pos_out = make_float4(pos.x + vel.x * dt, pos.y + vel.y * dt, pos.z + vel.z * dt, pos.w);
    vel_out = vel;
}

// Requires blockDim.x == BLOCK. Padding bodies in the last tile have mass 0.
__global__ void nbody_step(const float4* pos_in, const float4* vel_in, float4* pos_out,
                           float4* vel_out, int n, float dt) {
    __shared__ float4 tile[BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 my = (i < n) ? pos_in[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    for (int base = 0; base < n; base += BLOCK) {
        int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos_in[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncthreads();

        int count = min(BLOCK, n - base);
        for (int k = 0; k < count; ++k) acc = body_interaction(my, tile[k], acc);
        __syncthreads();
    }

    if (i < n) integrate(my, vel_in[i], acc, dt, pos_out[i], vel_out[i]);
}

static void cpu_step(const std::vector<float4>& pos, const std::vector<float4>& vel,
                     std::vector<float4>& pos_out, std::vector<float4>& vel_out, float dt) {
    int n = static_cast<int>(pos.size());
    for (int i = 0; i < n; ++i) {
        float3 acc = make_float3(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < n; ++j) acc = body_interaction(pos[i], pos[j], acc);
        integrate(pos[i], vel[i], acc, dt, pos_out[i], vel_out[i]);
    }
}

int main() {
    const int n = 4000;  // not a multiple of BLOCK on purpose
    const int steps = 4;
    const float dt = 0.01f;

    Lcg rng(7);
    std::vector<float4> pos(n), vel(n);
    for (int i = 0; i < n; ++i) {
        pos[i] = make_float4(rng.next_float(-1, 1), rng.next_float(-1, 1), rng.next_float(-1, 1),
                             rng.next_float(0.5f, 1.5f));
        vel[i] = make_float4(rng.next_float(-0.1f, 0.1f), rng.next_float(-0.1f, 0.1f),
                             rng.next_float(-0.1f, 0.1f), 0.0f);
    }

    std::vector<float4> ref_pos = pos, ref_vel = vel, tmp_pos(n), tmp_vel(n);
    for (int s = 0; s < steps; ++s) {
        cpu_step(ref_pos, ref_vel, tmp_pos, tmp_vel, dt);
        ref_pos.swap(tmp_pos);
        ref_vel.swap(tmp_vel);
    }

    DeviceBuffer<float4> d_pos0(n), d_vel0(n), d_pos1(n), d_vel1(n);
    int blocks = (n + BLOCK - 1) / BLOCK;

    report_time("nbody_step x4", time_ms([&] {
        d_pos0.upload(pos);
        d_vel0.upload(vel);
        for (int s = 0; s < steps; ++s) {
            bool even = (s % 2 == 0);
            nbody_step<<<blocks, BLOCK>>>(even ? d_pos0.get() : d_pos1.get(),
                                          even ? d_vel0.get() : d_vel1.get(),
                                          even ? d_pos1.get() : d_pos0.get(),
                                          even ? d_vel1.get() : d_vel0.get(), n, dt);
        }
    }));

    std::vector<float4> got_pos, got_vel;
    (steps % 2 == 0 ? d_pos0 : d_pos1).download(got_pos);
    (steps % 2 == 0 ? d_vel0 : d_vel1).download(got_vel);

    // rsqrtf is approximate on the GPU, hence the tolerance.
    bool ok = true;
    ok &= check_close(reinterpret_cast<const float*>(ref_pos.data()),
                      reinterpret_cast<const float*>(got_pos.data()), size_t(n) * 4, 1e-4, 1e-5,
                      "positions");
    ok &= check_close(reinterpret_cast<const float*>(ref_vel.data()),
                      reinterpret_cast<const float*>(got_vel.data()), size_t(n) * 4, 1e-4, 1e-4,
                      "velocities");
    return finish("nbody", ok);
}
