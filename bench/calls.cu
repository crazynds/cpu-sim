// Function calls that cannot be inlined: recursion, calls through function
// pointers and virtual calls. One source, two ways to run it:
//   cuda: kernels, one element per thread (the GPU with nvcc, the simulator
//         with cucc);
//   omp:  the same __host__ __device__ functions over the same elements, in a
//         C++ loop parallelized with OpenMP.
// Both print a checksum per workload, which must agree.
//
//   build/cucc -O3 -std=c++20 -fopenmp -Xcompiler -march=native bench/calls.cu -o build/bench/calls_cpu
//   nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp,-march=native bench/calls.cu -o build/bench/calls_gpu
//   build/bench/calls_cpu cuda|omp
#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>

#define CHECK(call)                                                                                     \
    do {                                                                                                \
        cudaError_t e_ = (call);                                                                        \
        if (e_ != cudaSuccess) {                                                                        \
            std::printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));          \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__host__ __device__ inline unsigned hash(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// ---------------------------------------------------------------- workloads

__host__ __device__ unsigned fib(unsigned n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

// Implicit binary search tree (children 2i+1, 2i+2): the depth of the key.
__host__ __device__ int bst_depth(const unsigned* tree, int n, int node, unsigned key) {
    if (node >= n) return -1;
    if (tree[node] == key) return 0;
    int d = bst_depth(tree, n, key < tree[node] ? 2 * node + 1 : 2 * node + 2, key);
    return d < 0 ? d : d + 1;
}

// Sum over a recursive subdivision of [lo, hi): the shape of a divide and
// conquer (merge sort, quadtree) without the memory.
__host__ __device__ float split_sum(unsigned seed, int lo, int hi) {
    if (hi - lo <= 2) return float(hash(seed + lo) & 1023) * (1.0f / 1024);
    int mid = (lo + hi) / 2;
    return split_sum(seed, lo, mid) + 0.5f * split_sum(seed, mid, hi);
}

typedef float (*unary_fn)(float);
__host__ __device__ float op_poly(float x) { return ((0.25f * x + 0.5f) * x + 1.0f) * x; }
__host__ __device__ float op_sqrt(float x) { return sqrtf(x + 1.0f); }
__host__ __device__ float op_recip(float x) { return 1.0f / (x + 1.0f); }
__host__ __device__ float op_mix(float x) { return x * 0.75f + 2.0f; }

__host__ __device__ inline float apply_ops(unsigned v, float x) {
    const unary_fn ops[4] = {op_poly, op_sqrt, op_recip, op_mix};
    for (int r = 0; r < 4; ++r) x = ops[(v >> (2 * r)) & 3](x);
    return x;
}

struct Shape {
    float a;
    __host__ __device__ explicit Shape(float a_) : a(a_) {}
    __host__ __device__ virtual float area() const = 0;
    __host__ __device__ virtual float perimeter() const = 0;
};
struct Square : Shape {
    __host__ __device__ explicit Square(float a_) : Shape(a_) {}
    __host__ __device__ float area() const override { return a * a; }
    __host__ __device__ float perimeter() const override { return 4.0f * a; }
};
struct Circle : Shape {
    __host__ __device__ explicit Circle(float a_) : Shape(a_) {}
    __host__ __device__ float area() const override { return 3.14159265f * a * a; }
    __host__ __device__ float perimeter() const override { return 6.2831853f * a; }
};
struct Hexagon : Shape {
    __host__ __device__ explicit Hexagon(float a_) : Shape(a_) {}
    __host__ __device__ float area() const override { return 2.5980762f * a * a; }
    __host__ __device__ float perimeter() const override { return 6.0f * a; }
};
union ShapeBox {
    Square s;
    Circle c;
    Hexagon h;
    __host__ __device__ ShapeBox() {}
};
// `sorted`: neighbours have the same type, as in an array grouped by type.
__host__ __device__ inline void make_shape(ShapeBox* box, unsigned i, bool sorted) {
    unsigned v = hash(i);
    float a = float(v % 100) * 0.01f;
    switch (sorted ? i / 4096 % 3 : v % 3) {
        case 0: new (box) Square(a); break;
        case 1: new (box) Circle(a); break;
        default: new (box) Hexagon(a); break;
    }
}
__host__ __device__ inline float shape_value(const ShapeBox* box) {
    const Shape* s = reinterpret_cast<const Shape*>(box);
    return s->area() + 0.5f * s->perimeter();
}

// ---------------------------------------------------------------- kernels

__global__ void k_fib(unsigned* out, int n, unsigned base, unsigned spread) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fib(base + (spread ? hash(i) % spread : 0));
}
__global__ void k_bst(const unsigned* tree, int tree_n, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = unsigned(bst_depth(tree, tree_n, 0, hash(i) % (2u * tree_n)));
}
__global__ void k_split(float* out, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = split_sum(hash(i), 0, width);
}
__global__ void k_ops(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = apply_ops(hash(i), float(i & 255) * 0.01f);
}
__global__ void k_make(ShapeBox* boxes, int n, bool sorted) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) make_shape(&boxes[i], unsigned(i), sorted);
}
__global__ void k_virtual(const ShapeBox* boxes, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = shape_value(&boxes[i]);
}

// ---------------------------------------------------------------- driver

struct Workload {
    const char* name;
    int n;  // elements
};

static double checksum_u(const unsigned* p, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += p[i] * double(1 + i % 7);
    return s;
}
static double checksum_f(const float* p, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += p[i] * double(1 + i % 7);
    return s;
}

// Best of `reps` runs of f, after one untimed run.
template <class F>
static double best_of(int reps, F f) {
    f();
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        f();
        best = std::min(best, now_s() - t0);
    }
    return best;
}

static void report(const char* name, int n, double seconds, double check) {
    std::printf("%-28s %10d  %10.3f  %10.1f  %.10g\n", name, n, seconds * 1e3, n / seconds / 1e6, check);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "cuda";
    if (mode != "cuda" && mode != "omp") {
        std::printf("usage: calls cuda|omp\n");
        return 1;
    }
    const bool cuda = mode == "cuda";
    const int fib_n = 1 << 15, bst_n = 1 << 23, tree_n = (1 << 22) - 1, split_n = 1 << 18, ops_n = 1 << 24,
              shape_n = 1 << 24;
    const int reps = 3;

    // The tree: every even number below 2 * tree_n, in order.
    std::vector<unsigned> tree(tree_n);
    {
        // In-order fill of the implicit tree without recursion: node ranks.
        std::vector<int> stack;
        unsigned next = 0;
        int node = 0;
        while (node < tree_n || !stack.empty()) {
            while (node < tree_n) {
                stack.push_back(node);
                node = 2 * node + 1;
            }
            node = stack.back();
            stack.pop_back();
            tree[node] = 2 * next++;
            node = 2 * node + 2;
        }
    }

    if (cuda) {
        cudaDeviceProp p;
        CHECK(cudaGetDeviceProperties(&p, 0));
        std::printf("cuda on %s\n", p.name);
        CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));  // a GPU's default (1 KiB) is too small
    } else {
        std::printf("omp, %d threads\n", omp_get_max_threads());
    }
    std::printf("%-28s %10s  %10s  %10s  %s\n", "workload", "elements", "time (ms)", "M elem/s", "checksum");

    const int block = 256;
    auto grid = [&](int n) { return (n + block - 1) / block; };
    std::vector<unsigned> hu(std::max(fib_n, bst_n));
    std::vector<float> hf(std::max(ops_n, shape_n));

    unsigned *du = nullptr, *dtree = nullptr;
    float* df = nullptr;
    ShapeBox* dbox = nullptr;
    std::unique_ptr<ShapeBox[]> hbox;
    if (cuda) {
        CHECK(cudaMalloc(&du, sizeof(unsigned) * std::max(fib_n, bst_n)));
        CHECK(cudaMalloc(&df, sizeof(float) * std::max(ops_n, shape_n)));
        CHECK(cudaMalloc(&dtree, sizeof(unsigned) * tree_n));
        CHECK(cudaMalloc(&dbox, sizeof(ShapeBox) * shape_n));
        CHECK(cudaMemcpy(dtree, tree.data(), sizeof(unsigned) * tree_n, cudaMemcpyHostToDevice));
    } else {
        hbox.reset(new ShapeBox[shape_n]);
    }
    auto sync = [] { CHECK(cudaDeviceSynchronize()); };

    // fib: about 1.6 calls per unit of the result; same n everywhere, then
    // n differing per element (the lanes of a warp recurse to different depths).
    for (unsigned spread : {0u, 6u}) {
        const unsigned base = spread ? 18 : 20;
        double t;
        if (cuda) {
            t = best_of(reps, [&] { k_fib<<<grid(fib_n), block>>>(du, fib_n, base, spread); sync(); });
            CHECK(cudaMemcpy(hu.data(), du, sizeof(unsigned) * fib_n, cudaMemcpyDeviceToHost));
        } else {
            t = best_of(reps, [&] {
#pragma omp parallel for schedule(dynamic, 64)
                for (int i = 0; i < fib_n; ++i) hu[i] = fib(base + (spread ? hash(i) % spread : 0));
            });
        }
        report(spread ? "fib, n per element 18..23" : "fib, n = 20", fib_n, t, checksum_u(hu.data(), fib_n));
    }

    {
        double t;
        if (cuda) {
            t = best_of(reps, [&] { k_bst<<<grid(bst_n), block>>>(dtree, tree_n, du, bst_n); sync(); });
            CHECK(cudaMemcpy(hu.data(), du, sizeof(unsigned) * bst_n, cudaMemcpyDeviceToHost));
        } else {
            t = best_of(reps, [&] {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < bst_n; ++i) hu[i] = unsigned(bst_depth(tree.data(), tree_n, 0, hash(i) % (2u * tree_n)));
            });
        }
        report("bst search, 4M nodes", bst_n, t, checksum_u(hu.data(), bst_n));
    }

    {
        const int width = 256;
        double t;
        if (cuda) {
            t = best_of(reps, [&] { k_split<<<grid(split_n), block>>>(df, split_n, width); sync(); });
            CHECK(cudaMemcpy(hf.data(), df, sizeof(float) * split_n, cudaMemcpyDeviceToHost));
        } else {
            t = best_of(reps, [&] {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < split_n; ++i) hf[i] = split_sum(hash(i), 0, width);
            });
        }
        report("divide and conquer, 256", split_n, t, checksum_f(hf.data(), split_n));
    }

    {
        double t;
        if (cuda) {
            t = best_of(reps, [&] { k_ops<<<grid(ops_n), block>>>(df, ops_n); sync(); });
            CHECK(cudaMemcpy(hf.data(), df, sizeof(float) * ops_n, cudaMemcpyDeviceToHost));
        } else {
            t = best_of(reps, [&] {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < ops_n; ++i) hf[i] = apply_ops(hash(i), float(i & 255) * 0.01f);
            });
        }
        report("function pointers, 4 per elem", ops_n, t, checksum_f(hf.data(), ops_n));
    }

    for (bool sorted : {true, false}) {
        double t;
        if (cuda) {
            k_make<<<grid(shape_n), block>>>(dbox, shape_n, sorted);
            sync();
            t = best_of(reps, [&] { k_virtual<<<grid(shape_n), block>>>(dbox, df, shape_n); sync(); });
            CHECK(cudaMemcpy(hf.data(), df, sizeof(float) * shape_n, cudaMemcpyDeviceToHost));
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < shape_n; ++i) make_shape(&hbox[i], unsigned(i), sorted);
            t = best_of(reps, [&] {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < shape_n; ++i) hf[i] = shape_value(&hbox[i]);
            });
        }
        report(sorted ? "virtual, grouped by type" : "virtual, mixed types", shape_n, t,
               checksum_f(hf.data(), shape_n));
    }
    return 0;
}
