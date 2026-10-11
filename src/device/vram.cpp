#include "device/vram.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

#include "common/units.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

namespace cpusim {

std::unique_ptr<VramPool> VramPool::create(uint64_t size, HugePages hugepages, bool prefault, uint64_t preferred_base,
                                           std::string* error) {
    std::unique_ptr<VramPool> pool(new VramPool);
    pool->hugepages_ = hugepages;
    pool->prefault_ = prefault;
    pool->page_size_ = hugepages == HugePages::OneGiB ? GiB : hugepages == HugePages::TwoMiB ? 2 * MiB : 4096;
    pool->size_ = (size + pool->page_size_ - 1) / pool->page_size_ * pool->page_size_;

    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (hugepages == HugePages::OneGiB) flags |= MAP_HUGETLB | (30 << MAP_HUGE_SHIFT);
    if (prefault) flags = (flags & ~MAP_NORESERVE) | MAP_POPULATE;

    void* p = MAP_FAILED;
    if (preferred_base != 0)
        p = mmap(reinterpret_cast<void*>(preferred_base), pool->size_, PROT_READ | PROT_WRITE,
                 flags | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) p = mmap(nullptr, pool->size_, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) {
        if (error) {
            *error = "mmap of " + format_size(pool->size_) + " failed: " + std::strerror(errno);
            if (hugepages == HugePages::OneGiB)
                *error += " (1GiB pages must be reserved first: hugepagesz=1G hugepages=N on the kernel command line)";
        }
        return nullptr;
    }
    pool->base_ = static_cast<uint8_t*>(p);

    // Transparent huge pages: needs "madvise" or "always" in
    // /sys/kernel/mm/transparent_hugepage/enabled; silently 4 KiB pages otherwise.
    if (hugepages == HugePages::TwoMiB) madvise(p, pool->size_, MADV_HUGEPAGE);
    return pool;
}

VramPool::~VramPool() {
    if (base_) munmap(base_, size_);
}

void VramPool::discard(uint64_t begin, uint64_t end) {
    if (prefault_) return;
    uint64_t b = (begin + page_size_ - 1) / page_size_ * page_size_;
    uint64_t e = end / page_size_ * page_size_;
    if (e > b) madvise(base_ + b, e - b, MADV_DONTNEED);
}

void VramPool::populate(uint64_t begin, uint64_t end) {
    if (prefault_ || end <= begin) return;
    uint64_t b = begin / page_size_ * page_size_;
    uint64_t e = std::min(size_, (end + page_size_ - 1) / page_size_ * page_size_);
#ifdef MADV_POPULATE_WRITE
    if (madvise(base_ + b, e - b, MADV_POPULATE_WRITE) == 0) return;
#endif
    // Older kernels: touch every page.
    for (uint64_t off = b; off < e; off += page_size_) *reinterpret_cast<volatile uint8_t*>(base_ + off) = 0;
}

uint64_t VramPool::resident_bytes() const {
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    const uint64_t chunk = 256 * MiB;
    std::vector<unsigned char> vec(chunk / page);
    uint64_t resident = 0;
    for (uint64_t off = 0; off < size_; off += chunk) {
        uint64_t len = std::min(chunk, size_ - off);
        if (mincore(base_ + off, len, vec.data()) != 0) return 0;
        for (uint64_t i = 0; i < (len + page - 1) / page; ++i) resident += vec[i] & 1;
    }
    return resident * page;
}

}  // namespace cpusim
