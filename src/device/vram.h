#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "common/config.h"

namespace cpusim {

// The device's memory: one anonymous mapping owned by whoever holds the device
// (the driver process, or the application with the in-process backend). Device
// pointers are addresses inside it.
class VramPool {
public:
    // `preferred_base` (0 = anywhere) is tried first, without replacing existing mappings.
    static std::unique_ptr<VramPool> create(uint64_t size, HugePages hugepages, bool prefault,
                                           uint64_t preferred_base, std::string* error);
    ~VramPool();

    VramPool(const VramPool&) = delete;
    VramPool& operator=(const VramPool&) = delete;

    uint8_t* base() const { return base_; }
    uint64_t size() const { return size_; }
    uint64_t page_size() const { return page_size_; }
    bool prefaulted() const { return prefault_; }
    HugePages hugepages() const { return hugepages_; }

    // Gives the whole pages inside [begin, end) (offsets) back to the OS, so freed
    // VRAM does not keep host RAM busy. No-op when the pool is prefaulted.
    void discard(uint64_t begin, uint64_t end);

    // Backs [begin, end) (offsets) with RAM now, as a GPU's cudaMalloc does:
    // kernels never stop on first-touch page faults (with 2 MiB pages, each
    // costs a 2 MiB clear in the middle of a launch).
    void populate(uint64_t begin, uint64_t end);

    // Bytes of the pool currently backed by host RAM.
    uint64_t resident_bytes() const;

private:
    VramPool() = default;

    uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    uint64_t page_size_ = 4096;
    bool prefault_ = false;
    HugePages hugepages_ = HugePages::None;
};

}  // namespace cpusim
