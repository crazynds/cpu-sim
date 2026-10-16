#include "device/width_tuning.h"

#include <algorithm>
#include <cstdio>
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
    cands_ = valid;
    samples_.assign(cands_.size(), {});
    schedule_.clear();
    for (size_t i = 0; i < cands_.size(); ++i) schedule_.push_back(i);
    pos_ = 0;
    round_ = 1;
}

namespace {
uint64_t median(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    return v[(v.size() - 1) / 2];  // the lower middle of an even count
}
}  // namespace

size_t WidthTuning::decide(const std::vector<std::vector<uint64_t>>& times) {
    std::vector<uint64_t> med(times.size(), UINT64_MAX);
    uint64_t best = UINT64_MAX;
    for (size_t i = 0; i < times.size(); ++i)
        if (!times[i].empty()) best = std::min(best, med[i] = median(times[i]));
    for (size_t i = 0; i < times.size(); ++i)
        if (med[i] != UINT64_MAX && double(med[i]) <= double(best) * (1.0 + kTie)) return i;
    return 0;
}

bool WidthTuning::record(uint32_t lanes, uint64_t ns, uint64_t work_ns) {
    if (chosen_ || pos_ >= schedule_.size() || cands_[schedule_[pos_]] != lanes) return false;
    if (!warm_) {
        warm_ = true;
        return false;
    }
    samples_[schedule_[pos_]].push_back(ns);
    if (round_ == 1) first_work_ = std::max(first_work_, work_ns);
    if (++pos_ < schedule_.size()) return false;
    // A round is done.
    if (round_ == 1 && first_work_ < kShortNs) {
        chosen_ = cands_[0];
        return true;
    }
    if (round_ == 1) {
        uint64_t best = UINT64_MAX;
        for (const auto& s : samples_) best = std::min(best, s.front());
        for (auto& s : samples_)
            if (double(s.front()) > double(best) * kDrop) s.clear();  // dropped
    }
    schedule_.clear();
    for (size_t i = 0; i < cands_.size(); ++i)
        if (!samples_[i].empty()) schedule_.push_back(i);
    pos_ = 0;
    if (++round_ <= kRounds && schedule_.size() > 1) return false;
    chosen_ = cands_[decide(samples_)];
    return true;
}

void WidthTuning::choose(const std::vector<uint32_t>& lanes, const std::vector<std::vector<uint64_t>>& slices) {
    cands_ = lanes;
    samples_ = slices;
    chosen_ = cands_[decide(samples_)];
}

std::vector<uint64_t> WidthTuning::medians() const {
    std::vector<uint64_t> m;
    for (const auto& s : samples_) m.push_back(s.empty() ? 0 : median(s));
    return m;
}

std::string WidthTuning::describe() const {
    std::string out;
    char buf[96];
    for (size_t i = 0; i < cands_.size(); ++i) {
        const auto& s = samples_[i];
        if (s.empty()) {
            std::snprintf(buf, sizeof(buf), " %u%s:-", cands_[i] & 0xffff, width_suffix(cands_[i]));
        } else {
            const auto [lo, hi] = std::minmax_element(s.begin(), s.end());
            std::snprintf(buf, sizeof(buf), " %u%s:%.1f[%.1f-%.1f]", cands_[i] & 0xffff, width_suffix(cands_[i]),
                          median(s) / 1e3, *lo / 1e3, *hi / 1e3);
        }
        out += buf;
    }
    return out;
}

uint32_t WidthDecisions::find(const Key& k) const {
    std::lock_guard lock(mutex_);
    auto it = map_.find(k);
    return it == map_.end() ? 0 : it->second;
}

void WidthDecisions::store(const Key& k, uint32_t lanes) {
    std::lock_guard lock(mutex_);
    if (map_.size() < kMaxEntries || map_.count(k)) map_[k] = lanes;
}

size_t WidthDecisions::size() const {
    std::lock_guard lock(mutex_);
    return map_.size();
}

}  // namespace cpusim
