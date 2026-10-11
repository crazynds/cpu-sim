#include <random>
#include <vector>

#include "check.h"
#include "device/allocator.h"

using cpusim::RangeAllocator;

int main() {
    constexpr uint64_t MiB = 1 << 20;
    {
        RangeAllocator a(64 * MiB);
        auto x = a.allocate(100);
        auto y = a.allocate(1000);
        EXPECT(x && y && *x % 256 == 0 && *y % 256 == 0 && *x != *y);
        EXPECT(a.used() == 256 + 1024);
        EXPECT(a.find(*y + 1000, 24).has_value());   // inside the rounded block
        EXPECT(!a.find(*y + 1000, 25).has_value());  // runs past the end
        EXPECT(a.release(*x) == 256);
        EXPECT(a.release(*x) == 0);                  // double free
        EXPECT(a.release(*y) == 1024);
        EXPECT(a.used() == 0 && a.largest_free() == 64 * MiB);
    }
    {
        // Large allocations start on 2 MiB boundaries.
        RangeAllocator a(64 * MiB);
        auto small = a.allocate(256);
        auto big = a.allocate(3 * MiB);
        EXPECT(small && big && *big % (2 * MiB) == 0);
        EXPECT(!a.allocate(64 * MiB).has_value());
        a.release(*small);
        a.release(*big);
        EXPECT(a.largest_free() == 64 * MiB);
    }
    {
        // Random churn: everything coalesces back into one free block.
        RangeAllocator a(256 * MiB);
        std::mt19937_64 rng(42);
        std::vector<uint64_t> live;
        for (int i = 0; i < 20000; ++i) {
            if (live.empty() || rng() % 3) {
                if (auto p = a.allocate(1 + rng() % (4 * MiB))) live.push_back(*p);
            } else {
                size_t k = rng() % live.size();
                EXPECT(a.release(live[k]) > 0);
                live[k] = live.back();
                live.pop_back();
            }
        }
        for (auto p : live) a.release(p);
        EXPECT(a.used() == 0 && a.allocation_count() == 0 && a.largest_free() == 256 * MiB);
    }
    TEST_MAIN_END();
}
