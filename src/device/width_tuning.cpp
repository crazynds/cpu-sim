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
// The mean of the times between the lowest eighth and the middle. What
// disturbs a launch (another process on the core, an interrupt, a page
// fault) only adds time, so the upper half is where the noise is; the lowest
// eighth is left out too, against a launch that measured unusually little.
uint64_t estimate(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    // (From 3 times on, at least the lowest is left out: the measurement may
    // stop after 3 rounds, and one lucky launch must not decide it.)
    const size_t lo = v.size() >= 3 ? std::max<size_t>(1, v.size() / 8) : 0;
    const size_t hi = std::max(lo + 1, (v.size() + 1) / 2);
    uint64_t sum = 0;
    for (size_t i = lo; i < hi; ++i) sum += v[i];
    return sum / (hi - lo);
}
}  // namespace

double WidthTuning::tie() {
    static const double t = [] {
        const char* v = std::getenv("CPUSIM_WIDTH_TIE");
        return v && std::atof(v) > 0 ? std::atof(v) : kTie;
    }();
    return t;
}

int WidthTuning::rounds() {
    static const int n = [] {
        const char* v = std::getenv("CPUSIM_WIDTH_ROUNDS");
        return v && std::atoi(v) > 0 ? std::atoi(v) : kRounds;
    }();
    return n;
}

namespace {
// The rule on estimates (0: dropped).
size_t decide_on(const std::vector<double>& est, double tie) {
    double best = 0;
    for (double e : est)
        if (e > 0 && (best == 0 || e < best)) best = e;
    for (size_t i = 0; i < est.size(); ++i)
        if (est[i] > 0 && est[i] <= best * (1.0 + tie)) return i;
    return 0;
}
}  // namespace

size_t WidthTuning::decide(const std::vector<std::vector<uint64_t>>& times) {
    std::vector<double> est(times.size(), 0);
    for (size_t i = 0; i < times.size(); ++i)
        if (!times[i].empty()) est[i] = double(estimate(times[i]));
    return decide_on(est, tie());
}

bool WidthTuning::settled(const std::vector<std::vector<uint64_t>>& times) {
    std::vector<double> est(times.size(), 0);
    for (size_t i = 0; i < times.size(); ++i)
        if (!times[i].empty()) est[i] = double(estimate(times[i]));
    const size_t d = decide_on(est, tie());
    // Every estimate kSettle off, up or down, in every combination.
    const size_t n = est.size();
    if (n > 12) return false;
    for (uint32_t mask = 0; mask < (1u << n); ++mask) {
        std::vector<double> e = est;
        for (size_t i = 0; i < n; ++i) e[i] *= (mask >> i & 1) ? 1.0 + kSettle : 1.0 - kSettle;
        if (decide_on(e, tie()) != d) return false;
    }
    return true;
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
        finish();
        return true;
    }
    if (round_ == 1) {
        uint64_t best = UINT64_MAX;
        for (const auto& s : samples_) best = std::min(best, s.front());
        // (Never the first candidate: it is what a tie falls back to, and a
        // dropped one would hand the tie to the next in one run only.)
        for (size_t i = 1; i < samples_.size(); ++i)
            if (double(samples_[i].front()) > double(best) * kDrop) samples_[i].clear();  // dropped
    }
    schedule_.clear();
    for (size_t i = 0; i < cands_.size(); ++i)
        if (!samples_[i].empty()) schedule_.push_back(i);
    pos_ = 0;
    // Settled early: the choice would be the same with every estimate kSettle
    // off either way (a clear winner needs kMinRounds, a near tie all of them).
    const bool early = round_ >= kMinRounds && settled(samples_);
    if (++round_ <= rounds() && schedule_.size() > 1 && !early) return false;
    chosen_ = cands_[decide(samples_)];
    finish();
    return true;
}

void WidthTuning::choose(const std::vector<uint32_t>& lanes, const std::vector<std::vector<uint64_t>>& slices) {
    cands_ = lanes;
    samples_ = slices;
    chosen_ = cands_[decide(samples_)];
    finish();
}

void WidthTuning::finish() {
    summary_ = describe();  // for messages
    samples_.clear();       // only the decision stays
    samples_.shrink_to_fit();
    schedule_.clear();
}

std::string WidthTuning::describe() const {
    if (chosen_ && !summary_.empty()) return summary_;
    std::string out;
    char buf[96];
    for (size_t i = 0; i < cands_.size() && i < samples_.size(); ++i) {
        const auto& s = samples_[i];
        if (s.empty()) {
            std::snprintf(buf, sizeof(buf), " %u%s:-", cands_[i] & 0xffff, width_suffix(cands_[i]));
        } else {
            const auto [lo, hi] = std::minmax_element(s.begin(), s.end());
            std::snprintf(buf, sizeof(buf), " %u%s:%.1f[%.1f-%.1f]", cands_[i] & 0xffff, width_suffix(cands_[i]),
                          estimate(s) / 1e3, *lo / 1e3, *hi / 1e3);
        }
        out += buf;
    }
    return out;
}

}  // namespace cpusim
