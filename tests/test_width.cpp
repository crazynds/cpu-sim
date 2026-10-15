// The width selection rules (device/width_tuning.h) with made-up times: which
// widths a launch may use and in what order, and which one the measurement
// keeps. No kernel runs here, so the result does not depend on the machine.
#include <algorithm>
#include <cstdint>
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
    // The fastest wins, and a candidate more than 1.5x slower than the best
    // of the first round is not measured again.
    int launches = 0;
    EXPECT(measure(W{32, 64, 128}, [](uint32_t w) { return w == 64 ? 300 : w == 128 ? 90 : 100; }, &launches) == 128);
    EXPECT(launches == 6);  // the warm-up, 3, and the two within 1.5x
    // Only one within 1.5x: chosen after the first round.
    EXPECT(measure(W{32, 64, 128}, [](uint32_t w) { return w == 32 ? 100 : 1000; }, &launches) == 32);
    EXPECT(launches == 4);
    // The first launch is cold (first touch of the data, caches): its time
    // does not count, so the width it runs is not dropped for it (HeCBench
    // adam: 32 lanes measured 2.9x slower than 128 in the first launch, and
    // was the fastest).
    {
        WidthTuning t;
        t.start(W{32, 64, 128});
        int launch = 0;
        while (!t.chosen()) {
            const uint32_t w = t.next();
            t.record(w, launch++ == 0 ? 300 : w == 32 ? 80 : 100);  // cold first launch
        }
        EXPECT(t.chosen() == 32);
    }
    // Each candidate keeps its best run: a slow first run (warm-up) does
    // not lose a width that is faster afterwards.
    {
        WidthTuning t;
        t.start(W{32, 64});
        t.record(32, 1000);  // the warm-up launch
        t.record(32, 100);
        t.record(64, 140);  // still warming up
        t.record(32, 100);
        t.record(64, 80);
        EXPECT(t.chosen() == 64);
    }
    // A time for a width other than the one asked for is ignored.
    {
        WidthTuning t;
        t.start(W{32, 64});
        t.record(32, 1);  // the warm-up launch
        EXPECT(!t.record(64, 1));
        EXPECT(t.next() == 32);
        EXPECT(!t.chosen());
    }
    // In-launch trials: the best slice wins.
    {
        WidthTuning t;
        t.start(W{256, 128 | kSuspend, 32});
        t.choose(W{256, 128 | kSuspend, 32}, {50, 40, 60});
        EXPECT(t.chosen() == (128 | kSuspend));
        EXPECT(!t.record(32, 1));  // nothing more once chosen
    }
    // Whatever the times, the choice is the fastest candidate within noise
    // of nothing: the one with the least time, over many made-up cases.
    unsigned seed = 1;
    auto rnd = [&] { return (seed = seed * 1103515245u + 12345u) >> 16; };
    for (int c = 0; c < 1000; ++c) {
        const W valid{32, 64, 128, 256};
        uint64_t cost[4];
        for (uint64_t& x : cost) x = 100 + rnd() % 1000;
        const uint32_t got = measure(valid, [&](uint32_t w) { return cost[w == 32 ? 0 : w == 64 ? 1 : w == 128 ? 2 : 3]; });
        uint64_t best = cost[0];
        for (uint64_t x : cost) best = std::min(best, x);
        const uint64_t mine = cost[got == 32 ? 0 : got == 64 ? 1 : got == 128 ? 2 : 3];
        EXPECT(mine == best);
    }
    TEST_MAIN_END();
}
