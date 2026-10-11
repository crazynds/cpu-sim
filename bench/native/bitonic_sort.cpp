// 240000 independent segments of 1024 ints sorted ascending
// (bench/bitonic_sort.cu); the bitonic network gives the same sorted segments.
#include <algorithm>

#include "native.hpp"

constexpr int SEG = 1024;  // SORT_N in samples/bitonic_sort.cu

int main() {
    const int segments = 240000;
    std::vector<int> data(size_t(segments) * SEG), work;
    Lcg rng(11);
    for (auto& v : data) v = int(rng.next_u32());
    native_report("bitonic_sort", native_ms(
                                      [&] {
#pragma omp parallel for schedule(static)
                                          for (int s = 0; s < segments; ++s)
                                              std::sort(work.begin() + size_t(s) * SEG,
                                                        work.begin() + size_t(s + 1) * SEG);
                                      },
                                      [&] { work = data; }));  // sort unsorted data every time
    native_checksum("bitonic_sort", work);
}
