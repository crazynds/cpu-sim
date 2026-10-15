#pragma once

// Width selection: which variants a launch may use, and the measurement that
// picks one per kernel and block shape. Nothing here runs or times a kernel:
// DeviceCore feeds in the times it measures, so the rules can be checked on
// their own (tests/test_width.cpp).

#include <cstdint>
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

// The measurement for one kernel and block shape. Two ways in:
//  - every candidate inside one large launch, on slices of its blocks
//    (choose(), the best slice of each);
//  - one candidate per launch, for smaller launches (next(), record()): a
//    first launch that only warms up (the kernel's first touch of its data
//    and cold caches made the first candidate look up to 3x slower, and it
//    was then dropped), a first round of every candidate, then a second one
//    of those within 1.5x of the best, keeping each candidate's best.
class WidthTuning {
public:
    static constexpr int kRounds = 2;

    bool started() const { return !candidates_.empty() || chosen_; }
    void start(const std::vector<uint32_t>& valid);
    uint32_t chosen() const { return chosen_; }
    // The width the next launch runs (only while nothing is chosen).
    uint32_t next() const { return candidates_[ns_.size()]; }
    // One launch's time per block with `lanes`. Ignored unless it is the
    // width next() asked for, and for the warm-up launch. True when this
    // time made the choice.
    bool record(uint32_t lanes, uint64_t ns);
    // In-launch trials: per candidate, the best of its slices.
    void choose(const std::vector<uint32_t>& lanes, const std::vector<uint64_t>& ns);

    // For messages: the first round's candidates and each one's best time
    // (valid once chosen through record()).
    const std::vector<uint32_t>& candidates() const { return candidates_; }
    const std::vector<uint64_t>& times() const { return ns_; }
    size_t first_round() const { return first_round_; }

private:
    std::vector<uint32_t> candidates_;  // widths still to run, round after round
    std::vector<uint64_t> ns_;          // one per run so far
    size_t first_round_ = 0;            // candidates in the first round, once it is done
    bool warm_ = false;                 // the warm-up launch is done
    uint32_t chosen_ = 0;
};

}  // namespace cpusim
