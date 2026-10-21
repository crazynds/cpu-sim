#pragma once

// Width selection: which variants a launch may use, and the measurement that
// picks one per kernel and block shape. Nothing here runs or times a kernel:
// DeviceCore feeds in the times it measures, so the rules can be checked on
// their own (tests/test_width.cpp).

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace cpusim {

// Widths are lane counts; kSuspend marks the 64/128-lane variants that still
// suspend at barriers (several groups per block, like 32), kRefill the warp
// entry whose lanes take another thread when theirs is done.
constexpr uint32_t kSuspend = 0x10000;
constexpr uint32_t kRefill = 0x40000;

// "s", "r" or "" (for messages: 64s, 32r, 128).
const char* width_suffix(uint32_t w);
// CPUSIM_LANES syntax: "<n>", "<n>s", "<n>r"; 0 when absent or not a number.
uint32_t parse_width(const char* text);

// The variants a kernel was compiled with (cpusim_kernel_meta and the entry
// points found in its module).
struct KernelVariants {
    uint32_t warp_lanes = 32;  // the warp entry's lanes (fewer for a block smaller than a warp)
    uint32_t warp_size = 32;
    bool barriers = false;
    bool susp[2] = {};  // suspending variants of 2W and 4W lanes
    bool wide[5] = {};  // variants of 2W << i lanes
    bool refill = false;
};

// The widths a launch with blocks of `threads` threads may use, in the order
// they are tried: 32 (warp by warp) and the wider variants valid for it. A
// kernel with barriers may only use a variant whose width covers the whole
// block (the smallest such one), or a suspending one narrower than the block;
// one without barriers can use any up to the block. With barriers the wide
// variant comes first and the warp entry last: the warp entry, which suspends
// at every barrier, has been the slowest every time we measured, and a
// program too short for the measurement to finish keeps the first ones.
std::vector<uint32_t> valid_widths(const KernelVariants& v, uint64_t threads);

// The measurement for one kernel and block shape, made so that the same
// program picks the same width every time (a daemon restarted must not decide
// otherwise). Two ways in:
//  - every candidate inside one large launch, on slices of its blocks, in
//    kRounds rounds (choose());
//  - one candidate per launch, for smaller launches (next(), record()): a
//    first launch that only warms up (the kernel's first touch of its data
//    and cold caches made the first candidate look up to 3x slower), then
//    kRounds rounds of every candidate, interleaved so that a drift of the
//    clock or of the machine's load hits all of them alike. After the first
//    round, a candidate more than kDrop times slower than the best is not run
//    again (each run is a whole launch; a 20x slower one costs 20 launches).
// Short launches are not measured: when the first round's launches did less
// than kShortNs of work in all (the workers' time summed), the first
// candidate in the order of valid_widths is chosen. There the launch's fixed
// cost outweighs the width, the time a worker takes is mostly noise, and the
// executor calls a number of workers it learns from launch to launch, so
// the candidates did not even run alike (a 12 us kernel picked 128 lanes on
// one daemon and 64 on the next).
// The decision: each candidate's estimate (the mean of its times between the
// lowest eighth and the middle: what disturbs a launch only adds time), and
// among the candidates within kTie of the best, the first in the order of
// valid_widths, which is never dropped. Widths that close are a tie the noise would
// decide one way in one run and the other in the next; the order decides
// them the same way every time, and only a clear difference is measured.
class WidthTuning {
public:
    // (Single launches measured up to 2x off the rest of their candidate's,
    // so only 3x drops a candidate. On 5 fresh drivers running 49 programs,
    // 5 rounds and a 10% tie left 17 of 53 kernel shapes with more than one
    // width, 15 rounds 12 of 49, and 15 rounds with a 25% tie 4 of 49: the
    // estimates were steady by then, and the kernels that alternated had two
    // widths whose real gap was close to the tie margin.)
    static constexpr int kRounds = 15;
    // In-launch trials: slices of 32 blocks per worker, long and steady;
    // 15 of them would need launches of half a million blocks.
    static constexpr int kSliceRounds = 5;
    static constexpr double kDrop = 3.0;
    static constexpr double kTie = 0.25;
    // The measurement stops before kRounds once at least kMinRounds are done
    // and the choice would be the same with every estimate kSettle off either
    // way: a clear winner (about 1.9x with these margins) is decided in 3
    // rounds, so a program of a few launches measures 3 rounds, not 15; a
    // closer call gets all 15. (With 10%, estimates from 3 rounds settled
    // wrong now and then: 8 of 56 kernel shapes alternated on fresh drivers.)
    static constexpr int kMinRounds = 3;
    static constexpr double kSettle = 0.20;
    static int rounds();  // kRounds, or CPUSIM_WIDTH_ROUNDS
    static double tie();  // kTie, or CPUSIM_WIDTH_TIE
    static constexpr uint64_t kShortNs = 1000000;

    bool started() const { return !cands_.empty() || chosen_; }
    void start(const std::vector<uint32_t>& valid);
    uint32_t chosen() const { return chosen_; }
    // The width the next launch runs (only while nothing is chosen).
    uint32_t next() const { return cands_[schedule_[pos_]]; }
    // One launch's time per block with `lanes`, and its work (the workers'
    // time summed). Ignored unless it is the width next() asked for, and for
    // the warm-up launch. True when this launch made the choice.
    bool record(uint32_t lanes, uint64_t ns, uint64_t work_ns = UINT64_MAX);
    // In-launch trials: each candidate's slice times, one per round.
    void choose(const std::vector<uint32_t>& lanes, const std::vector<std::vector<uint64_t>>& slices);

    // For messages: the candidates and each one's estimate so far, and
    // all its times ("100.2 [91.0-130.5]", us per 1000 blocks); once
    // decided, as they were then.
    const std::vector<uint32_t>& candidates() const { return cands_; }
    std::string describe() const;
    // Chosen by the rule for short launches, not by the times.
    bool short_launches() const { return chosen_ && round_ == 1 && first_work_ < kShortNs; }

    // The rule itself: the chosen index, given each candidate's times
    // (empty: dropped), in the order of preference.
    static size_t decide(const std::vector<std::vector<uint64_t>>& times);
    // Whether decide() would choose the same with every estimate kSettle off.
    static bool settled(const std::vector<std::vector<uint64_t>>& times);

private:
    std::vector<uint32_t> cands_;                 // in the order of valid_widths
    std::vector<std::vector<uint64_t>> samples_;  // per candidate
    std::vector<size_t> schedule_;                // this round's candidates (indices)
    size_t pos_ = 0;                              // the next in schedule_
    int round_ = 0;
    bool warm_ = false;  // the warm-up launch is done
    uint64_t first_work_ = 0;  // the most work of a launch in the first round
    std::string summary_;      // the times, once decided (describe())
    void finish();             // decided: drops the times
    uint32_t chosen_ = 0;
};

// What a measurement is for: one kernel's code (a hash of its module and its
// name), a block shape, and the number of workers. The daemon keeps one
// measurement per key while it runs, across programs: a program too short to
// finish it (one launch, ten) leaves it where the next program that runs the
// same code picks it up, and once it is decided only the width remains (the
// times are dropped). In memory only, gone with the daemon.
struct WidthKey {
    uint64_t code = 0, shape = 0;
    uint32_t workers = 0;
    bool operator<(const WidthKey& o) const {
        return std::tie(code, shape, workers) < std::tie(o.code, o.shape, o.workers);
    }
};

}  // namespace cpusim
