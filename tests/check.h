#pragma once

#include <cstdio>
#include <cstdlib>

static int g_failures = 0;

#define EXPECT(cond)                                                          \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "%s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

#define TEST_MAIN_END()                                                       \
    do {                                                                      \
        std::printf("%s\n", g_failures ? "FAIL" : "PASS");                   \
        return g_failures ? 1 : 0;                                            \
    } while (0)
