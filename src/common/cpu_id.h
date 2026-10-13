#pragma once

// A signature of what this CPU can run: the instruction set extensions it
// reports (CPUID) and the register state the OS saves for programs (XCR0:
// AVX-512, AMX). Code compiled for "this machine" (--mcpu=native) records it;
// the runtime uses that code only on a machine with the same signature, and
// compiles from the bitcode (JIT) on any other, so a program built on a CPU
// with more extensions (AVX512-FP16, AMX) never runs their instructions on
// one without. Equal signatures, not compatible ones: two different CPUs
// compile again, which costs a JIT, never a crash.

#include <cpuid.h>

#include <cstdint>

namespace cpusim {

inline uint64_t cpu_signature() {
    static const uint64_t sig = [] {
        uint64_t h = 1469598103934665603ull;  // FNV-1a over the registers
        auto mix = [&](uint32_t v) {
            for (int i = 0; i < 4; ++i) h = (h ^ ((v >> (8 * i)) & 0xffu)) * 1099511628211ull;
        };
        unsigned a, b, c, d;
        if (__get_cpuid(0, &a, &b, &c, &d)) mix(b), mix(c), mix(d);  // vendor
        if (__get_cpuid(1, &a, &b, &c, &d)) mix(a & 0x0fff0ff0u), mix(c), mix(d);  // family/model, features
        const unsigned max = __get_cpuid_max(0, nullptr);
        if (max >= 7) {
            __cpuid_count(7, 0, a, b, c, d);
            mix(b), mix(c), mix(d);
            __cpuid_count(7, 1, a, b, c, d);
            mix(a), mix(d);
        }
        if (__get_cpuid(0x80000001, &a, &b, &c, &d)) mix(c), mix(d);
        if (__get_cpuid(1, &a, &b, &c, &d) && (c >> 27 & 1)) {  // OSXSAVE: XCR0 readable
            uint32_t lo, hi;
            __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            mix(lo), mix(hi);
        }
        return h;
    }();
    return sig;
}

}  // namespace cpusim
