// Device functions that cannot be inlined: recursion (direct, mutual, deep,
// with per-lane depths), calls through function pointers (a table indexed per
// lane, a pointer the same for the whole block) and virtual functions on
// objects built by another kernel. Results must match the host exactly.
#include <cstdint>
#include <new>
#include <vector>

#include "check.cuh"

__host__ __device__ unsigned fib(unsigned n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

__host__ __device__ bool is_odd(unsigned n);
__host__ __device__ bool is_even(unsigned n) { return n == 0 ? true : is_odd(n - 1); }
__host__ __device__ bool is_odd(unsigned n) { return n == 0 ? false : is_even(n - 1); }

__host__ __device__ unsigned ackermann(unsigned m, unsigned n) {
    if (m == 0) return n + 1;
    if (n == 0) return ackermann(m - 1, 1);
    return ackermann(m - 1, ackermann(m, n - 1));
}

// Binary search tree stored as an implicit array (children 2i+1, 2i+2):
// depth of the key, or -1.
__host__ __device__ int bst_depth(const int* tree, int n, int node, int key) {
    if (node >= n) return -1;
    if (tree[node] == key) return 0;
    int d = bst_depth(tree, n, key < tree[node] ? 2 * node + 1 : 2 * node + 2, key);
    return d < 0 ? d : d + 1;
}

// Writes through its pointer at every level, returns nothing.
__host__ __device__ void countdown(unsigned* out, unsigned n) {
    if (n == 0) return;
    *out += n;
    countdown(out, n - 1);
}

__host__ __device__ double power_rec(double x, int n) {
    if (n == 0) return 1.0;
    double h = power_rec(x, n / 2);
    return n % 2 ? h * h * x : h * h;
}

__global__ void recursion(const unsigned* in, const int* tree, int tree_n, unsigned* out, double* dout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = in[i];
    unsigned r = fib(v % 20);
    r = r * 3 + (is_even(v % 300) ? 1 : 0);
    r = r * 7 + ackermann(v % 3, v % 5);
    r = r * 11 + unsigned(bst_depth(tree, tree_n, 0, int(v % 200)));
    unsigned acc = 0;
    if (v & 1) countdown(&out[n + i], v % 50);
    out[i] = r;
    dout[i] = power_rec(1.0 + (v % 7) * 0.125, int(v % 40));
    (void)acc;
}

// Recursion in a kernel with barriers (the call is between them).
__global__ void recursion_barrier(const unsigned* in, unsigned* out, int n) {
    __shared__ unsigned s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = i < n ? fib(in[i] % 16) : 0;
    __syncthreads();
    unsigned other = s[(threadIdx.x + 1) % blockDim.x];
    __syncthreads();
    if (i < n) out[i] = other + fib(other % 12);
}

typedef float (*unary_fn)(float);
__host__ __device__ float op_square(float x) { return x * x; }
__host__ __device__ float op_half(float x) { return x * 0.5f; }
__host__ __device__ float op_neg(float x) { return -x; }
__host__ __device__ float op_plus3(float x) { return x + 3.0f; }
static unary_fn host_ops[4] = {op_square, op_half, op_neg, op_plus3};

__global__ void pointers(const unsigned* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unary_fn device_ops[4] = {op_square, op_half, op_neg, op_plus3};
    unsigned v = in[i];
    float x = float(v % 1000) * 0.25f;
    float r = device_ops[v % 4](x);           // per lane
    r = device_ops[blockIdx.x % 4](r);        // the same for the block
    if (v % 3 == 0) r = device_ops[(v / 4) % 4](r);  // some lanes only
    out[i] = r;
}

struct Shape {
    float a;
    __host__ __device__ explicit Shape(float a_) : a(a_) {}
    __host__ __device__ virtual float area() const = 0;
    __host__ __device__ virtual float scaled(float k) const { return k * area(); }
};
struct Square : Shape {
    __host__ __device__ explicit Square(float a_) : Shape(a_) {}
    __host__ __device__ float area() const override { return a * a; }
};
struct Circle : Shape {
    __host__ __device__ explicit Circle(float a_) : Shape(a_) {}
    __host__ __device__ float area() const override { return 3.0f * a * a; }
    __host__ __device__ float scaled(float k) const override { return k + area(); }
};
struct Tri : Shape {
    __host__ __device__ explicit Tri(float a_) : Shape(a_) {}
    __host__ __device__ float area() const override { return 0.5f * a * a; }
};
union ShapeBox {
    Square s;
    Circle c;
    Tri t;
    __host__ __device__ ShapeBox() {}
};

__host__ __device__ Shape* make_shape(ShapeBox* box, unsigned v) {
    float a = float(v % 100) * 0.5f;
    switch (v % 3) {
        case 0: return new (box) Square(a);
        case 1: return new (box) Circle(a);
        default: return new (box) Tri(a);
    }
}

__global__ void build_shapes(const unsigned* in, ShapeBox* boxes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) make_shape(&boxes[i], in[i]);
}

__global__ void use_shapes(const ShapeBox* boxes, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Shape* s = reinterpret_cast<const Shape*>(&boxes[i]);
    out[i] = s->scaled(float(i % 5)) + s->area();
}

int main() {
    Checker check("calls");
    const int n = 10000;
    std::vector<unsigned> in(n);
    uint64_t state = 12345;
    for (auto& v : in) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        v = unsigned(state >> 33);
    }
    // A balanced BST of the even numbers 0..198 in implicit layout.
    const int tree_n = 100;
    std::vector<int> tree(tree_n);
    {
        int next = 0;
        auto fill = [&](auto&& self, int node) -> void {
            if (node >= tree_n) return;
            self(self, 2 * node + 1);
            tree[node] = 2 * next++;
            self(self, 2 * node + 2);
        };
        fill(fill, 0);
    }

    unsigned *d_in, *d_out;
    int* d_tree;
    double* d_dout;
    float* d_fout;
    ShapeBox* d_boxes;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc(&d_out, 2 * n * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc(&d_tree, tree_n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dout, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_fout, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_boxes, n * sizeof(ShapeBox)));
    CHECK_CUDA(cudaMemcpy(d_in, in.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tree, tree.data(), tree_n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, 2 * n * sizeof(unsigned)));

    std::vector<unsigned> out(2 * n);
    std::vector<double> dout(n);
    std::vector<float> fout(n);

    for (int block : {32, 96, 256}) {
        CHECK_CUDA(cudaMemset(d_out, 0, 2 * n * sizeof(unsigned)));
        recursion<<<(n + block - 1) / block, block>>>(d_in, d_tree, tree_n, d_out, d_dout, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(out.data(), d_out, 2 * n * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(dout.data(), d_dout, n * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            unsigned v = in[i];
            unsigned r = fib(v % 20);
            r = r * 3 + (is_even(v % 300) ? 1 : 0);
            r = r * 7 + ackermann(v % 3, v % 5);
            r = r * 11 + unsigned(bst_depth(tree.data(), tree_n, 0, int(v % 200)));
            unsigned c = 0;
            if (v & 1) countdown(&c, v % 50);
            check.expect(out[i] == r, "recursion", i, out[i], r);
            check.expect(out[n + i] == c, "recursion (void, through a pointer)", i, out[n + i], c);
            double p = power_rec(1.0 + (v % 7) * 0.125, int(v % 40));
            check.expect(dout[i] == p, "recursion (double)", i, dout[i], p);
        }

        recursion_barrier<<<(n + block - 1) / block, block>>>(d_in, d_out, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(out.data(), d_out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            int b0 = i / block * block, t = i - b0, j = b0 + (t + 1) % block;
            unsigned other = j < n ? fib(in[j] % 16) : 0;
            unsigned want = other + fib(other % 12);
            check.expect(out[i] == want, "recursion between barriers", i, out[i], want);
        }

        pointers<<<(n + block - 1) / block, block>>>(d_in, d_fout, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(fout.data(), d_fout, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            unsigned v = in[i];
            float r = host_ops[v % 4](float(v % 1000) * 0.25f);
            r = host_ops[(i / block) % 4](r);
            if (v % 3 == 0) r = host_ops[(v / 4) % 4](r);
            check.expect(fout[i] == r, "function pointers", i, fout[i], r);
        }

        build_shapes<<<(n + block - 1) / block, block>>>(d_in, d_boxes, n);
        use_shapes<<<(n + block - 1) / block, block>>>(d_boxes, d_fout, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(fout.data(), d_fout, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            ShapeBox box;
            const Shape* s = make_shape(&box, in[i]);
            float want = s->scaled(float(i % 5)) + s->area();
            check.expect(fout[i] == want, "virtual functions", i, fout[i], want);
        }
    }
    return check.finish();
}
