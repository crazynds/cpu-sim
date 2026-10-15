#include "device/width_tuning.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace cpusim {

const char* width_suffix(uint32_t w) { return w & kSuspend ? "s" : w & kRefill ? "r" : ""; }

uint32_t parse_width(const char* text) {
    if (!text) return 0;
    const uint32_t w = uint32_t(std::atoi(text));
    if (!w) return 0;
    return std::strchr(text, 's') ? w | kSuspend : std::strchr(text, 'r') ? w | kRefill : w;
}

std::vector<uint32_t> valid_widths(const KernelVariants& v, uint64_t threads) {
    const uint32_t warp = v.warp_lanes;
    const uint32_t two = 2 * v.warp_size;  // the narrowest wide variant
    std::vector<uint32_t> valid{warp};
    for (int i = 0; i < 2; ++i)
        if (v.susp[i] && (two << i) < threads) valid.push_back(two << i | kSuspend);
    for (int i = 0; i < 5; ++i) {
        const uint32_t w = two << i;
        if (!v.wide[i]) continue;
        if (v.barriers) {
            if (threads <= w) {
                valid.push_back(w);
                break;  // the smallest covering width is enough
            }
        } else if (w <= threads) {
            valid.push_back(w);
        }
    }
    // Lanes that refill: for warps whose lanes loop for different counts.
    if (v.refill) valid.push_back(warp | kRefill);
    if (v.barriers) std::stable_sort(valid.begin(), valid.end(), [&](uint32_t a, uint32_t b) {
        auto rank = [&](uint32_t w) { return w == warp ? 3 : (w & kSuspend) ? 2 : 0; };
        return rank(a) < rank(b) || (rank(a) == rank(b) && (a & 0xffff) > (b & 0xffff));
    });
    return valid;
}

void WidthTuning::start(const std::vector<uint32_t>& valid) {
    // Every candidate once per round, keeping each one's best: warm-up and
    // clock changes hit all of them alike.
    for (int r = 0; r < kRounds; ++r) candidates_.insert(candidates_.end(), valid.begin(), valid.end());
}

bool WidthTuning::record(uint32_t lanes, uint64_t ns) {
    if (chosen_ || ns_.size() >= candidates_.size() || candidates_[ns_.size()] != lanes) return false;
    if (!warm_) {
        warm_ = true;
        return false;
    }
    ns_.push_back(ns);
    const size_t n = first_round_ ? first_round_ : candidates_.size() / kRounds;
    if (!first_round_ && ns_.size() == n) {
        // First round done. The second one only separates candidates within
        // 1.5x of the best; the others cannot win and are not run again
        // (each run is a whole launch with that width, and a program that
        // launches a shape a few times would spend them all measuring).
        first_round_ = n;
        const uint64_t best = *std::min_element(ns_.begin(), ns_.end());
        candidates_.resize(n);
        for (int r = 1; r < kRounds; ++r)
            for (size_t i = 0; i < n; ++i)
                if (ns_[i] <= best + best / 2) candidates_.push_back(candidates_[i]);
        if (candidates_.size() == n + kRounds - 1) candidates_.resize(n);  // only one left: no need
    }
    if (ns_.size() != candidates_.size()) return false;
    // Each candidate's best over its runs.
    std::vector<uint64_t> best_ns(ns_.begin(), ns_.begin() + n);
    for (size_t k = n; k < ns_.size(); ++k)
        for (size_t i = 0; i < n; ++i)
            if (candidates_[i] == candidates_[k]) best_ns[i] = std::min(best_ns[i], ns_[k]);
    size_t best = 0;
    for (size_t i = 0; i < n; ++i) {
        ns_[i] = best_ns[i];
        if (ns_[i] < ns_[best]) best = i;
    }
    chosen_ = candidates_[best];
    return true;
}

void WidthTuning::choose(const std::vector<uint32_t>& lanes, const std::vector<uint64_t>& ns) {
    const size_t best = size_t(std::min_element(ns.begin(), ns.end()) - ns.begin());
    chosen_ = lanes[best];
}

}  // namespace cpusim
