#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>

namespace cpusim {

// Best-fit range allocator over [0, size). Knows nothing about memory itself:
// the VRAM pool maps offsets to addresses. Not thread safe.
class RangeAllocator {
public:
    // Every allocation is a multiple of `granularity` and aligned to it (cudaMalloc
    // guarantees 256 bytes). Allocations of at least `large_threshold` bytes are
    // aligned to `large_alignment` so they start on a huge page.
    RangeAllocator(uint64_t size, uint64_t granularity = 256, uint64_t large_threshold = 2 << 20,
                   uint64_t large_alignment = 2 << 20);

    std::optional<uint64_t> allocate(uint64_t bytes);

    // Frees the allocation that starts exactly at `offset` and returns its size
    // (0 if there is none). [*free_begin, *free_end) receives the coalesced free
    // range around it, so the caller can give whole pages back to the OS.
    uint64_t release(uint64_t offset, uint64_t* free_begin = nullptr, uint64_t* free_end = nullptr);

    // Start and size of the allocation containing [offset, offset + len), if any.
    std::optional<std::pair<uint64_t, uint64_t>> find(uint64_t offset, uint64_t len) const;

    uint64_t size() const { return size_; }
    uint64_t used() const { return used_; }
    size_t allocation_count() const { return allocated_.size(); }
    uint64_t largest_free() const;
    // Allocations of 2 GiB or more: while there is none, an offset inside any
    // allocation fits in 32 signed bits (kernels then gather with 32-bit indices).
    static constexpr uint64_t kHuge = uint64_t(1) << 31;
    size_t huge_allocations() const { return huge_; }

private:
    void insert_free(uint64_t off, uint64_t len);
    void erase_free(std::map<uint64_t, uint64_t>::iterator it);

    uint64_t size_, granularity_, large_threshold_, large_alignment_;
    uint64_t used_ = 0;
    size_t huge_ = 0;
    std::map<uint64_t, uint64_t> free_by_offset_;              // offset -> length
    std::set<std::pair<uint64_t, uint64_t>> free_by_size_;     // (length, offset)
    std::map<uint64_t, uint64_t> allocated_;                   // offset -> length
};

}  // namespace cpusim
