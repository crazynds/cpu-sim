// Aggregate types in device memory: bitfields, unions (type punning and
// tagged), arrays of structs with 12- and 20-byte elements (no vector
// alignment), structs of arrays, nested structs returned by value, alignas.
#include "stress.cuh"

// Bitfields packed into one 32-bit word, signed and unsigned fields.
struct Packed {
    unsigned kind : 3;
    int delta : 7;  // signed, -64..63
    unsigned count : 10;
    unsigned flag : 1;
    unsigned rest : 11;
};
static_assert(sizeof(Packed) == 4, "packed");
__host__ __device__ inline Packed update(Packed p, int i) {
    p.kind = p.kind + 3;           // wraps modulo 8
    p.delta = p.delta - (i & 15);  // wraps in 7 bits (implementation-defined, two's complement)
    p.count = p.count * 3 + 1;     // wraps modulo 1024
    p.flag = !p.flag;
    p.rest ^= unsigned(i);
    return p;
}
__host__ __device__ inline int score(Packed p) { return int(p.kind) * 1000003 + p.delta * 1009 + int(p.count) * 7 + int(p.flag) + int(p.rest) * 13; }
__global__ void k_bitfield(Packed* data, int* scores, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Packed p = data[i];
    for (int k = 0; k < 5; ++k) p = update(p, i + k);
    data[i] = p;
    scores[i] = score(p);
}

// Union: the bits of a float and of an int; a tagged union.
union FloatBits {
    float f;
    unsigned u;
    unsigned char b[4];
};
struct Tagged {
    int tag;  // 0: int, 1: float, 2: two shorts
    union {
        int i;
        float f;
        short s[2];
    };
};
__host__ __device__ inline float tagged_value(const Tagged& t) {
    switch (t.tag) {
        case 0: return float(t.i);
        case 1: return t.f;
        default: return float(t.s[0]) - float(t.s[1]);
    }
}
__global__ void k_union(const float* in, unsigned* bits, const Tagged* tg, float* vals, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    FloatBits fb;
    fb.f = in[i];
    fb.b[3] ^= 0x80;  // flip the sign through the byte view (little endian)
    bits[i] = fb.u + fb.b[0];
    vals[i] = tagged_value(tg[i]);
}

// 12-byte AoS element (3 floats) and 20-byte element (mixed).
struct P3 {
    float x, y, z;
};
static_assert(sizeof(P3) == 12, "12");
struct Rec20 {
    int id;
    float w;
    short a, b;
    float v[2];
};
static_assert(sizeof(Rec20) == 20, "20");
__host__ __device__ inline P3 cross(P3 a, P3 b) {
    return P3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__global__ void k_aos12(const P3* a, const P3* b, P3* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cross(a[i], b[(i + 1) % n]);
}
__host__ __device__ inline Rec20 step(Rec20 r, const Rec20& o) {
    r.id = r.id * 31 + o.id;
    r.w = r.w * 0.5f + o.w;
    r.a = short(r.a + o.b);
    r.b = short(r.b ^ o.a);
    r.v[0] = r.v[0] - o.v[1];
    r.v[1] = r.v[1] + o.v[0];
    return r;
}
__global__ void k_aos20(const Rec20* in, Rec20* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Rec20 r = in[i];
    for (int k = 1; k <= 3; ++k) r = step(r, in[(i + k * 97) % n]);
    out[i] = r;
}

// SoA of the same data as the 12-byte AoS: identical results expected.
struct SoA3 {
    const float *x, *y, *z;
};
__global__ void k_soa12(SoA3 a, SoA3 b, float* ox, float* oy, float* oz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = (i + 1) % n;
    P3 c = cross(P3{a.x[i], a.y[i], a.z[i]}, P3{b.x[j], b.y[j], b.z[j]});
    ox[i] = c.x, oy[i] = c.y, oz[i] = c.z;
}

// Nested structs returned by value; alignas(16) element.
struct alignas(16) Body {
    struct {
        float x, y;
    } pos;
    struct {
        short hp;
        unsigned char level, team;
    } stats;
    int id;
};
static_assert(sizeof(Body) == 16 && alignof(Body) == 16, "16");
__host__ __device__ inline Body evolve(Body b, int t) {
    Body r = b;
    r.pos.x = b.pos.x + float(t) * 0.25f;
    r.pos.y = b.pos.y - float(b.stats.level);
    r.stats.hp = short(b.stats.hp - b.stats.team * t);
    r.stats.level = (unsigned char)(b.stats.level + 1);
    r.id = b.id ^ t;
    return r;
}
__global__ void k_nested(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Body b = bodies[i];
    for (int t = 0; t < 4; ++t) b = evolve(b, t + i % 3);
    bodies[i] = b;
}

template <class T>
static bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            std::printf("  mismatch at element %zu\n", i);
            return false;
        }
    return true;
}

int main() {
    using namespace stress;
    const int n = 1 << 18, B = 128, G = (n + B - 1) / B;
    auto r = random_u32(n, 5);

    {
        std::vector<Packed> d(n), want;
        for (int i = 0; i < n; ++i) std::memcpy(&d[i], &r[i], 4);
        std::vector<int> ws(n);
        double ref = best_ms([&] {
            want = d;
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                Packed p = want[i];
                for (int k = 0; k < 5; ++k) p = update(p, i + k);
                want[i] = p;
                ws[i] = score(p);
            }
        });
        Buffer<Packed> dd(d);
        Buffer<int> dsc(n);
        double k = time_device([&] {
            dd.upload(d);
            k_bitfield<<<G, B>>>(dd, dsc, n);
        });
        report("bitfields", same_bytes(dd.download(), want) && equal(dsc.download().data(), ws.data(), n), k, ref);
    }
    {
        auto f = random_floats(n, 6, -1e3f, 1e3f);
        std::vector<Tagged> tg(n);
        for (int i = 0; i < n; ++i) {
            tg[i].tag = r[i] % 3;
            if (tg[i].tag == 0) tg[i].i = int(r[i] >> 4) - (1 << 27);
            else if (tg[i].tag == 1) tg[i].f = f[(i + 9) % n];
            else tg[i].s[0] = short(r[i] >> 3), tg[i].s[1] = short(r[i] >> 17);
        }
        std::vector<unsigned> wb(n);
        std::vector<float> wv(n);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                FloatBits fb;
                fb.f = f[i];
                fb.b[3] ^= 0x80;
                wb[i] = fb.u + fb.b[0];
                wv[i] = tagged_value(tg[i]);
            }
        });
        Buffer<float> df(f), dv(n);
        Buffer<unsigned> dbits(n);
        Buffer<Tagged> dt(tg);
        double k = time_device([&] { k_union<<<G, B>>>(df, dbits, dt, dv, n); });
        report("unions_punning_tagged", equal(dbits.download().data(), wb.data(), n) && equal(dv.download().data(), wv.data(), n), k, ref);
    }

    auto fa = random_floats(3 * size_t(n), 7), fb = random_floats(3 * size_t(n), 8);
    std::vector<P3> a(n), b(n);
    for (int i = 0; i < n; ++i) a[i] = P3{fa[3 * i], fa[3 * i + 1], fa[3 * i + 2]}, b[i] = P3{fb[3 * i], fb[3 * i + 1], fb[3 * i + 2]};
    std::vector<P3> want3(n);
    double ref3 = best_ms([&] {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) want3[i] = cross(a[i], b[(i + 1) % n]);
    });
    // a*b - c*d may be contracted to fma on either side: compare with a tolerance
    // relative to the magnitude of the products (inputs in [-1,1]).
    auto near3 = [&](const P3& g, const P3& w) {
        return std::fabs(g.x - w.x) <= 3e-7f && std::fabs(g.y - w.y) <= 3e-7f && std::fabs(g.z - w.z) <= 3e-7f;
    };
    {
        Buffer<P3> da(a), db(b), dout(n);
        double k = time_device([&] { k_aos12<<<G, B>>>(da, db, dout, n); });
        auto got = dout.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (!near3(got[i], want3[i])) ok = false, std::printf("  mismatch at %d\n", i);
        report("aos_12_byte_cross", ok, k, ref3);
    }
    {
        std::vector<float> ax(n), ay(n), az(n), bx(n), by(n), bz(n);
        for (int i = 0; i < n; ++i) ax[i] = a[i].x, ay[i] = a[i].y, az[i] = a[i].z, bx[i] = b[i].x, by[i] = b[i].y, bz[i] = b[i].z;
        Buffer<float> dax(ax), day(ay), daz(az), dbx(bx), dby(by), dbz(bz), ox(n), oy(n), oz(n);
        double k = time_device([&] { k_soa12<<<G, B>>>(SoA3{dax, day, daz}, SoA3{dbx, dby, dbz}, ox, oy, oz, n); });
        auto gx = ox.download(), gy = oy.download(), gz = oz.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            if (!near3(P3{gx[i], gy[i], gz[i]}, want3[i])) ok = false, std::printf("  mismatch at %d\n", i);
        report("soa_12_byte_cross", ok, k, ref3);
    }
    {
        std::vector<Rec20> in(n), want(n);
        for (int i = 0; i < n; ++i)
            in[i] = Rec20{int(r[i]), float(r[i] % 1000) * 0.125f, short(r[i] >> 3), short(r[i] >> 19), {fa[i], fb[i]}};
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                Rec20 x = in[i];
                for (int k = 1; k <= 3; ++k) x = step(x, in[(i + k * 97) % n]);
                want[i] = x;
            }
        });
        Buffer<Rec20> din(in), dout(n);
        double k = time_device([&] { k_aos20<<<G, B>>>(din, dout, n); });
        report("aos_20_byte_records", same_bytes(dout.download(), want), k, ref);
    }
    {
        std::vector<Body> in(n), want;
        for (int i = 0; i < n; ++i) {
            in[i] = Body{};
            in[i].pos.x = fa[i] * 100, in[i].pos.y = fb[i] * 100;
            in[i].stats.hp = short(r[i] % 30000), in[i].stats.level = (unsigned char)(r[i] >> 8), in[i].stats.team = (unsigned char)(r[i] >> 16) % 4;
            in[i].id = i;
        }
        double ref = best_ms([&] {
            want = in;
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                for (int t = 0; t < 4; ++t) want[i] = evolve(want[i], t + i % 3);
        });
        Buffer<Body> dbod(in);
        double k = time_device([&] {
            dbod.upload(in);
            k_nested<<<G, B>>>(dbod, n);
        });
        auto got = dbod.download();
        bool ok = true;
        for (int i = 0; i < n && ok; ++i)
            ok = got[i].pos.x == want[i].pos.x && got[i].pos.y == want[i].pos.y && got[i].stats.hp == want[i].stats.hp &&
                 got[i].stats.level == want[i].stats.level && got[i].stats.team == want[i].stats.team && got[i].id == want[i].id;
        report("nested_struct_alignas16", ok, k, ref);
    }
    return finish();
}
