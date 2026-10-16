// The width selection rules (device/width_tuning.h) with made-up times: which
// widths a launch may use and in what order, and which one the measurement
// keeps. No kernel runs here, so the result does not depend on the machine.
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "check.h"
#include "device/width_tuning.h"

using namespace cpusim;
using W = std::vector<uint32_t>;

static KernelVariants all_variants(bool barriers) {
    KernelVariants v;
    v.barriers = barriers;
    v.susp[0] = v.susp[1] = barriers;  // only kernels with barriers have them
    for (int i = 0; i < 4; ++i) v.wide[i] = true;  // 64 to 512 (CPUSIM_MAX_LANES)
    return v;
}

// Runs the one-candidate-per-launch measurement: each launch takes `time(w)`
// per block. Returns the chosen width and how many launches measuring took.
template <class Time>
static uint32_t measure(const W& valid, Time time, int* launches = nullptr) {
    WidthTuning t;
    t.start(valid);
    int n = 0;
    while (!t.chosen() && n < 100) {
        const uint32_t w = t.next();
        t.record(w, time(w));
        ++n;
    }
    if (launches) *launches = n;
    return t.chosen();
}

int main() {
    // --- which widths, in which order
    EXPECT(parse_width("64") == 64);
    EXPECT(parse_width("64s") == (64 | kSuspend));
    EXPECT(parse_width("32r") == (32 | kRefill));
    EXPECT(parse_width("auto") == 0);
    EXPECT(parse_width(nullptr) == 0);

    // Without barriers: every wide variant up to the block, in width order.
    EXPECT(valid_widths(all_variants(false), 256) == (W{32, 64, 128, 256}));
    EXPECT(valid_widths(all_variants(false), 100) == (W{32, 64}));
    // With barriers: the smallest wide variant that covers the block first,
    // then the suspending ones narrower than the block, widest first, then 32.
    EXPECT(valid_widths(all_variants(true), 256) == (W{256, 128 | kSuspend, 64 | kSuspend, 32}));
    EXPECT(valid_widths(all_variants(true), 100) == (W{128, 64 | kSuspend, 32}));
    EXPECT(valid_widths(all_variants(true), 64) == (W{64, 32}));
    EXPECT(valid_widths(all_variants(true), 1024) == (W{128 | kSuspend, 64 | kSuspend, 32}));
    // A variant that was not built is never offered.
    KernelVariants some = all_variants(true);
    some.wide[2] = false;  // no 256
    EXPECT(valid_widths(some, 256) == (W{512, 128 | kSuspend, 64 | kSuspend, 32}));
    // A block smaller than a warp: only the narrow warp entry.
    KernelVariants small = all_variants(false);
    small.warp_lanes = 16;
    EXPECT(valid_widths(small, 16) == (W{16}));
    // Refill comes last.
    KernelVariants refill = all_variants(false);
    refill.refill = true;
    EXPECT(valid_widths(refill, 64) == (W{32, 64, 32 | kRefill}));
    // Warps of 16: widths scale with the warp.
    KernelVariants w16 = all_variants(false);
    w16.warp_size = w16.warp_lanes = 16;
    EXPECT(valid_widths(w16, 64) == (W{16, 32, 64}));

    // --- which one the measurement keeps
    // Launches: the warm-up, then 5 rounds; a candidate more than 3x slower
    // than the best after the first round is not run again.
    int launches = 0;
    EXPECT(measure(W{32, 64, 128}, [](uint32_t w) { return w == 64 ? 300 : w == 128 ? 70 : 100; }, &launches) == 128);
    EXPECT(launches == 1 + 3 + 4 * 2);  // 64 dropped after the first round
    // Only one left after the first round: chosen there.
    EXPECT(measure(W{32, 64, 128}, [](uint32_t w) { return w == 32 ? 100 : 1000; }, &launches) == 32);
    EXPECT(launches == 1 + 3);
    // The first launch is cold (first touch of the data, caches): its time
    // does not count, so the width it runs is not dropped for it (HeCBench
    // adam: 32 lanes measured 2.9x slower than 128 in the first launch, and
    // was the fastest).
    {
        int launch = 0;
        EXPECT(measure(W{32, 64, 128}, [&](uint32_t w) { return launch++ == 0 ? 300 : w == 32 ? 70 : 100; }) == 32);
    }
    // A candidate 2x off in its first run is not dropped (outliers that
    // large happen; 3x drops it), and wins if it is the faster one.
    EXPECT(measure(W{32, 64}, [n = 0](uint32_t w) mutable { return w == 64 && n++ == 0 ? 190 : 90 + (w == 32) * 10; }) ==
           64);
    // The median of each candidate's runs: two slow (or lucky) launches do
    // not decide.
    {
        WidthTuning t;
        t.start(W{32, 64});
        const uint64_t times[] = {999,        // warm-up
                                  100, 95,    // round 1
                                  100, 40,    // round 2: 64 lucky
                                  100, 96,    // round 3
                                  100, 41,    // round 4: lucky again
                                  100, 97};   // round 5
        for (uint64_t ns : times) t.record(t.next(), ns);
        EXPECT(t.chosen() == 32);  // medians: 32 -> 100, 64 -> 95, a tie: the first
    }
    // Within 10% of the best is a tie, decided by the order of valid_widths
    // (the same way in every run); a clear difference decides by the times.
    EXPECT(WidthTuning::decide({{105}, {100}}) == 0);
    EXPECT(WidthTuning::decide({{100}, {105}}) == 0);
    EXPECT(WidthTuning::decide({{120}, {100}}) == 1);
    EXPECT(WidthTuning::decide({{}, {300}, {100}}) == 2);  // dropped candidates do not count
    // The same program decides the same way however its times wobble: the
    // times of two widths 5% apart, each launch off by up to +-2%, never decide
    // differently from one run to the next (the tie rule), and two widths 40%
    // apart always go to the faster one.
    unsigned seed = 7;
    auto rnd = [&] { return (seed = seed * 1103515245u + 12345u) >> 16; };
    auto noisy = [&](uint64_t base) { return base * (980 + rnd() % 41) / 1000; };
    bool same = true, fast = true;
    for (int run = 0; run < 1000; ++run) {
        same &= measure(W{32, 64}, [&](uint32_t w) { return noisy(w == 32 ? 105 : 100); }) == 32;
        fast &= measure(W{32, 64}, [&](uint32_t w) { return noisy(w == 32 ? 140 : 100); }) == 64;
    }
    EXPECT(same);
    EXPECT(fast);
    // A time for a width other than the one asked for is ignored.
    {
        WidthTuning t;
        t.start(W{32, 64});
        t.record(32, 1);  // the warm-up launch
        EXPECT(!t.record(64, 1));
        EXPECT(t.next() == 32);
        EXPECT(!t.chosen());
    }
    // In-launch trials: the median of each candidate's slices, the same tie rule.
    {
        WidthTuning t;
        t.choose(W{256, 128 | kSuspend, 32}, {{50, 52, 51}, {40, 41, 39}, {60, 61, 62}});
        EXPECT(t.chosen() == (128 | kSuspend));
        EXPECT(!t.record(32, 1));  // nothing more once chosen
        t.choose(W{256, 128 | kSuspend, 32}, {{42, 43, 41}, {40, 41, 39}, {60, 61, 62}});
        EXPECT(t.chosen() == 256);  // within 10% of the best: the first
    }

    // Short launches (under 1 ms of work in the first round): the first
    // candidate, whatever the times, after one round.
    {
        WidthTuning t;
        t.start(W{32, 64, 128});
        int n = 0;
        while (!t.chosen()) {
            const uint32_t w = t.next();
            t.record(w, w == 128 ? 10 : 30, 400000);  // 0.4 ms of work
            ++n;
        }
        EXPECT(t.chosen() == 32 && t.short_launches());
        EXPECT(n == 1 + 3);
        // One launch of the first round over 1 ms: measured as usual.
        WidthTuning u;
        u.start(W{32, 64, 128});
        while (!u.chosen()) {
            const uint32_t w = u.next();
            u.record(w, w == 128 ? 10 : 30, w == 64 ? 1500000 : 400000);
        }
        EXPECT(u.chosen() == 128 && !u.short_launches());
    }

    // --- the measurement is kept per code, shape and workers, across
    // programs; once decided only the width (and its summary) remain
    {
        const WidthKey a{0x1234, (uint64_t(256) << 40) | (1 << 20) | 1, 62};
        const WidthKey b{0x1234, (uint64_t(128) << 40) | (1 << 20) | 1, 62};  // another shape
        const WidthKey c{0x1234, a.shape, 30};                                // other workers
        const WidthKey d{0x5678, a.shape, 62};                                // other code
        EXPECT(a < b || b < a);
        EXPECT(a < c || c < a);
        EXPECT(a < d || d < a);
        EXPECT(!(a < a));
        // A program of one launch leaves the measurement where the next one
        // picks it up: one launch per program.
        WidthTuning t;
        t.start(W{32, 64});
        int programs = 0;
        while (!t.chosen()) {
            const uint32_t w = t.next();  // this program's only launch
            t.record(w, w == 64 ? 60 : 100);
            ++programs;
        }
        EXPECT(t.chosen() == 64 && programs == 1 + 2 * 5);
        const std::string summary = t.describe();
        EXPECT(summary.find("64:0.1") != std::string::npos);  // the times as they were
    }
    TEST_MAIN_END();
}
