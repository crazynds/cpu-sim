#include "device/allocator.h"

#include <algorithm>

namespace cpusim {

namespace {

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

}  // namespace

RangeAllocator::RangeAllocator(uint64_t size, uint64_t granularity, uint64_t large_threshold,
                               uint64_t large_alignment)
    : size_(size / granularity * granularity),
      granularity_(granularity),
      large_threshold_(large_threshold),
      large_alignment_(large_alignment) {
    if (size_ > 0) insert_free(0, size_);
}

void RangeAllocator::insert_free(uint64_t off, uint64_t len) {
    free_by_offset_.emplace(off, len);
    free_by_size_.emplace(len, off);
}

void RangeAllocator::erase_free(std::map<uint64_t, uint64_t>::iterator it) {
    free_by_size_.erase({it->second, it->first});
    free_by_offset_.erase(it);
}

std::optional<uint64_t> RangeAllocator::allocate(uint64_t bytes) {
    if (bytes == 0 || bytes > size_) return std::nullopt;
    bytes = align_up(bytes, granularity_);
    uint64_t align = bytes >= large_threshold_ ? large_alignment_ : granularity_;

    // Allocations of 2 GiB or more go at the top of the highest free block
    // that holds them: the kernels index every other object with 32-bit
    // offsets, and test only whether a pointer falls in the span of these.
    if (bytes >= kHuge) {
        for (auto it = free_by_offset_.rbegin(); it != free_by_offset_.rend(); ++it) {
            auto [off, len] = *it;
            if (len < bytes) continue;
            uint64_t start = (off + len - bytes) / align * align;
            if (start < off) continue;
            erase_free(free_by_offset_.find(off));
            if (start > off) insert_free(off, start - off);
            if (off + len > start + bytes) insert_free(start + bytes, off + len - start - bytes);
            allocated_.emplace(start, bytes);
            huge_.emplace(start, bytes);
            used_ += bytes;
            return start;
        }
        return std::nullopt;
    }

    // Smallest free block that fits once its start is aligned.
    for (auto it = free_by_size_.lower_bound({bytes, 0}); it != free_by_size_.end(); ++it) {
        auto [len, off] = *it;
        uint64_t start = align_up(off, align);
        if (start - off + bytes > len) continue;

        erase_free(free_by_offset_.find(off));
        if (start > off) insert_free(off, start - off);
        uint64_t end = start + bytes;
        if (off + len > end) insert_free(end, off + len - end);

        allocated_.emplace(start, bytes);
        used_ += bytes;
        return start;
    }
    return std::nullopt;
}

uint64_t RangeAllocator::release(uint64_t offset, uint64_t* free_begin, uint64_t* free_end) {
    auto a = allocated_.find(offset);
    if (a == allocated_.end()) return 0;
    uint64_t len = a->second;
    allocated_.erase(a);
    used_ -= len;
    huge_.erase(offset);

    uint64_t begin = offset, end = offset + len;
    auto next = free_by_offset_.lower_bound(end);
    if (next != free_by_offset_.end() && next->first == end) {
        end += next->second;
        erase_free(next);
    }
    auto prev = free_by_offset_.lower_bound(begin);
    if (prev != free_by_offset_.begin()) {
        --prev;
        if (prev->first + prev->second == begin) {
            begin = prev->first;
            erase_free(prev);
        }
    }
    insert_free(begin, end - begin);
    if (free_begin) *free_begin = begin;
    if (free_end) *free_end = end;
    return len;
}

std::optional<std::pair<uint64_t, uint64_t>> RangeAllocator::find(uint64_t offset, uint64_t len) const {
    auto it = allocated_.upper_bound(offset);
    if (it == allocated_.begin()) return std::nullopt;
    --it;
    uint64_t start = it->first, n = it->second;
    if (offset - start > n || len > n - (offset - start)) return std::nullopt;
    if (len == 0 && offset == start + n) return std::nullopt;
    return std::make_pair(start, n);
}

std::optional<std::pair<uint64_t, uint64_t>> RangeAllocator::huge_span() const {
    if (huge_.empty()) return std::nullopt;
    uint64_t end = 0;
    for (auto [off, len] : huge_) end = std::max(end, off + len);
    return std::make_pair(huge_.begin()->first, end);
}

uint64_t RangeAllocator::largest_free() const {
    return free_by_size_.empty() ? 0 : free_by_size_.rbegin()->first;
}

}  // namespace cpusim
