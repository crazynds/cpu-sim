// Divergent control flow without loops at its core: if/else nested 8 deep, a
// 16-case switch with fallthrough, goto, a recursion-free state machine (a
// lexer), a bytecode interpreter dispatching through a switch on loaded
// opcodes, short-circuit evaluation with side effects, ternary chains.
#include "stress.cuh"

// 8 levels of if/else on bits of the input: every lane takes its own path.
__host__ __device__ inline int nested_if(unsigned v, int x) {
    if (v & 1) {
        x += 3;
        if (v & 2) {
            x *= 5;
            if (v & 4) {
                x ^= 0x55;
                if (v & 8) {
                    x -= 7;
                    if (v & 16) {
                        x = x * 3 + 1;
                        if (v & 32) {
                            x >>= 1;
                            if (v & 64) {
                                x += 1000;
                                if (v & 128) x = -x;
                                else x += 11;
                            } else {
                                x -= 1000;
                            }
                        } else {
                            x <<= 2;
                        }
                    } else if (v & 32) {
                        x += 17;
                    } else {
                        x -= 17;
                    }
                } else {
                    x = ~x;
                }
            } else if (v & 8) {
                if (v & 16) x += 2;
                else x -= 2;
            } else {
                x *= 7;
            }
        } else {
            x -= 9;
            if (v & 4) {
                if (v & 8) {
                    if (v & 16) {
                        if (v & 32) x += 64;
                        else x += 32;
                    } else {
                        x += 16;
                    }
                } else {
                    x += 8;
                }
            }
        }
    } else {
        if (v & 2) {
            x = x * 2 - 1;
            if (!(v & 4)) {
                if (v & 8) {
                    x |= 0x100;
                    if (v & 16) {
                        if (v & 32) {
                            if (v & 64) x &= 0xfff;
                            else x |= 0x1000;
                        }
                    }
                }
            }
        } else if (v & 4) {
            x += int(v >> 8) % 13;
        } else {
            x = 42;
        }
    }
    return x;
}
__global__ void k_nested(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = nested_if(in[i], int(in[i] >> 12) & 0xffff);
}

// 16-case switch with fallthrough chains and a default for values out of range.
__host__ __device__ inline int sw16(int k, int x) {
    int r = 0;
    switch (k) {
        case 0: r += 1; [[fallthrough]];
        case 1: r += x; break;
        case 2: r = x * 2; break;
        case 3: r = x * x; [[fallthrough]];
        case 4: r -= 5; [[fallthrough]];
        case 5: r ^= 0xabc; break;
        case 6: r = x >> 3; break;
        case 7: r = -x; break;
        case 8:
        case 9:
        case 10: r = x % 17 + k; break;
        case 11: r = x & 0xff; [[fallthrough]];
        case 12: r += 12; [[fallthrough]];
        case 13: r += 13; [[fallthrough]];
        case 14: r *= 3; break;
        case 15: return x + 1515;
        default: r = -1; break;
    }
    return r + k;
}
__global__ void k_switch(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = in[i];
    out[i] = sw16(int(v % 18) - 1, int(v >> 10) & 0x3fff);
}

// goto: a retry loop and a shared error exit.
__host__ __device__ inline int with_goto(unsigned v) {
    int tries = 0, acc = int(v & 0xff);
retry:
    acc = acc * 3 + 1;
    if (acc % 7 == 0) goto fail;
    if (++tries < int((v >> 8) & 7)) goto retry;
    if (acc & 1) goto odd;
    return acc;
odd:
    return acc + 100000;
fail:
    return -tries;
}
__global__ void k_goto(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = with_goto(in[i]);
}

// A lexer state machine over a per-thread string of 48 bytes: counts numbers,
// identifiers, operators, and sums the numbers.
enum State { START, NUMBER, IDENT, OP, DONE };
__host__ __device__ inline int lex(const unsigned char* s, int len) {
    State st = START;
    int pos = 0, nums = 0, ids = 0, ops = 0, value = 0, sum = 0;
    while (st != DONE) {
        unsigned char c = pos < len ? s[pos] : 0;
        switch (st) {
            case START:
                if (c == 0) st = DONE;
                else if (c >= '0' && c <= '9') st = NUMBER, value = 0;
                else if ((c >= 'a' && c <= 'z') || c == '_') st = IDENT;
                else if (c == '+' || c == '-' || c == '*') st = OP;
                else ++pos;  // whitespace and anything else
                break;
            case NUMBER:
                if (c >= '0' && c <= '9') value = value * 10 + (c - '0'), ++pos;
                else sum += value, ++nums, st = START;
                break;
            case IDENT:
                if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') ++pos;
                else ++ids, st = START;
                break;
            case OP:
                ++ops, ++pos, st = START;
                break;
            case DONE: break;
        }
    }
    return sum * 4096 + nums * 256 + ids * 16 + ops;
}
__global__ void k_lex(const unsigned char* text, const int* lens, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lex(text + 48 * i, lens[i]);
}

// A stack-machine interpreter: each thread runs its own program of 32 opcodes.
enum Opc : unsigned char { PUSH, ADD, SUB, MUL, DUP, SWAP, NEG, JZ, JMP, XOR, SHR, HALT };
__host__ __device__ inline int interp(const unsigned char* code, int x) {
    int st[16], sp = 0, pc = 0, steps = 0;
    st[sp++] = x;
    while (pc < 32 && steps++ < 200) {
        unsigned char op = code[pc] % 12, arg = code[pc] >> 4;
        ++pc;
        switch (op) {
            case PUSH: if (sp < 16) st[sp++] = arg; break;
            case ADD: if (sp > 1) { st[sp - 2] += st[sp - 1]; --sp; } break;
            case SUB: if (sp > 1) { st[sp - 2] -= st[sp - 1]; --sp; } break;
            case MUL: if (sp > 1) { st[sp - 2] = int(unsigned(st[sp - 2]) * unsigned(st[sp - 1])); --sp; } break;
            case DUP: if (sp > 0 && sp < 16) { st[sp] = st[sp - 1]; ++sp; } break;
            case SWAP: if (sp > 1) { int t = st[sp - 1]; st[sp - 1] = st[sp - 2]; st[sp - 2] = t; } break;
            case NEG: if (sp > 0) st[sp - 1] = int(0u - unsigned(st[sp - 1])); break;
            case JZ: if (sp > 0 && st[--sp] == 0) pc = arg * 2; break;
            case JMP: pc = (pc + arg) % 32; break;
            case XOR: if (sp > 1) { st[sp - 2] ^= st[sp - 1]; --sp; } break;
            case SHR: if (sp > 0) st[sp - 1] = int(unsigned(st[sp - 1]) >> (arg & 7)); break;
            case HALT: pc = 32; break;
        }
    }
    int r = sp * 1000003;
    for (int k = 0; k < sp; ++k) r ^= st[k] * (k + 1);
    return r;
}
__global__ void k_interp(const unsigned char* code, const int* x, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = interp(code + 32 * i, x[i]);
}

// Short-circuit evaluation: the right operand runs (and counts) only when needed.
__host__ __device__ inline bool probe(int v, int bit, int* count) {
    ++*count;
    return (v >> bit) & 1;
}
__host__ __device__ inline int short_circuit(int v) {
    int count = 0, r = 0;
    if (probe(v, 0, &count) && probe(v, 1, &count) && probe(v, 2, &count)) r += 1;
    if (probe(v, 3, &count) || probe(v, 4, &count) || probe(v, 5, &count)) r += 2;
    if ((probe(v, 6, &count) && !probe(v, 7, &count)) || probe(v, 8, &count)) r += 4;
    r += (v & 512) ? (v & 1024 ? 8 : (v & 2048 ? 16 : 24)) : (v & 4096 ? 32 : 40);
    return r * 100 + count;
}
__global__ void k_short(const unsigned* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = short_circuit(int(in[i]));
}

int main() {
    using namespace stress;
    const int n = 1 << 18, B = 256, G = (n + B - 1) / B;
    auto r = random_u32(n, 1);
    Buffer<unsigned> din(r);

    auto host = [&](auto f) {
        std::vector<int> w(n);
        double t = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) w[i] = f(i);
        });
        return std::make_pair(w, t);
    };
    auto check = [&](const char* name, auto f, auto launch) {
        auto [want, ref] = host(f);
        Buffer<int> dout(n);
        double k = time_device([&] { launch(dout.p); });
        report(name, equal(dout.download().data(), want.data(), n), k, ref);
    };

    check("nested_if_8_levels", [&](int i) { return nested_if(r[i], int(r[i] >> 12) & 0xffff); },
          [&](int* o) { k_nested<<<G, B>>>(din, o, n); });
    check("switch_16_fallthrough", [&](int i) { return sw16(int(r[i] % 18) - 1, int(r[i] >> 10) & 0x3fff); },
          [&](int* o) { k_switch<<<G, B>>>(din, o, n); });
    check("goto_retry_and_exits", [&](int i) { return with_goto(r[i]); }, [&](int* o) { k_goto<<<G, B>>>(din, o, n); });
    check("short_circuit_side_effects", [&](int i) { return short_circuit(int(r[i])); }, [&](int* o) { k_short<<<G, B>>>(din, o, n); });

    {
        const int m = 1 << 16;
        const char alphabet[] = "0123456789abcxyz_+-* \t";
        std::vector<unsigned char> text(48 * size_t(m));
        std::vector<int> lens(m);
        auto t = random_u32(48 * size_t(m), 2);
        for (int i = 0; i < m; ++i) {
            lens[i] = int(t[i] % 49);
            for (int k = 0; k < 48; ++k) text[48 * i + k] = alphabet[t[48 * i + k] % 22];
        }
        std::vector<int> want(m);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < m; ++i) want[i] = lex(text.data() + 48 * i, lens[i]);
        });
        Buffer<unsigned char> dt(text);
        Buffer<int> dl(lens), dout(m);
        double k = time_device([&] { k_lex<<<m / 128, 128>>>(dt, dl, dout, m); });
        report("state_machine_lexer", equal(dout.download().data(), want.data(), m), k, ref);
    }
    {
        const int m = 1 << 16;
        auto c = random_u32(32 * size_t(m), 3);
        std::vector<unsigned char> code(32 * size_t(m));
        for (size_t k = 0; k < code.size(); ++k) code[k] = (unsigned char)c[k];
        std::vector<int> x(m), want(m);
        for (int i = 0; i < m; ++i) x[i] = int(r[i] % 1000);
        double ref = best_ms([&] {
#pragma omp parallel for
            for (int i = 0; i < m; ++i) want[i] = interp(code.data() + 32 * i, x[i]);
        });
        Buffer<unsigned char> dc(code);
        Buffer<int> dx(x), dout(m);
        double k = time_device([&] { k_interp<<<m / 64, 64>>>(dc, dx, dout, m); });
        report("bytecode_interpreter_switch", equal(dout.download().data(), want.data(), m), k, ref);
    }
    return finish();
}
