#pragma once

// Waiting without sleeping, for a while. On the path of a kernel launch every
// thread that sleeps costs a system call and a wake-up of several
// microseconds (the program's stream thread, the driver's connection thread,
// the reply, the workers); a launch that follows another closely finds them
// all awake instead.
//
// Two budgets. The workers, pinned one per CPU, spin up to CPUSIM_SPIN_US
// (default 200 us). Every other thread is not pinned and may share a CPU with
// a worker, which it then slows down (the scheduler splits the CPU between
// them, yield or not): it spins only CPUSIM_WAIT_SPIN_US (default 20 us),
// enough for a short kernel's round trip, then sleeps. 0 turns either off.

#include <sched.h>
#include <sys/socket.h>

#include <immintrin.h>
#include <x86intrin.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace cpusim {

inline uint64_t spin_ns() {  // pinned workers
    static const uint64_t ns = [] {
        const char* v = std::getenv("CPUSIM_SPIN_US");
        return uint64_t(v ? std::atof(v) * 1000 : 200000);
    }();
    return ns;
}

inline uint64_t wait_spin_ns() {  // every other thread
    static const uint64_t ns = [] {
        const char* v = std::getenv("CPUSIM_WAIT_SPIN_US");
        return uint64_t(v ? std::atof(v) * 1000 : 20000);
    }();
    return ns;
}

// MONITORX/MWAITX (AMD, user mode): a waiting thread parks on a cache line
// until it is written or a timeout, without taking execution resources from
// the SMT sibling the way a pause loop does. CPUID 0x80000001, ECX bit 29.
inline bool has_mwaitx() {
    static const bool yes = [] {
        unsigned a = 0x80000001, b = 0, c = 0, d = 0;
        __asm__("cpuid" : "+a"(a), "=b"(b), "=c"(c), "=d"(d) : "c"(0));
        return (c >> 29) & 1;
    }();
    return yes;
}

// One short wait for a write to `line` (at most ~`ticks` TSC ticks).
__attribute__((target("mwaitx"))) inline void park_on(const void* line, unsigned ticks) {
    _mm_monitorx(const_cast<void*>(line), 0, 0);
    _mm_mwaitx(2 /* timer */, 0, ticks);
}

// Spins until ready() holds or the time is up; returns ready()'s last answer.
// For threads that are not pinned (the program's, the driver's connection
// threads): after a few microseconds of pause, each check gives the CPU away
// (sched_yield), so a kernel worker sharing it runs first instead of being
// held up by a thread that only waits.
template <class Ready>
bool spin_until(Ready ready) {
    if (ready()) return true;
    const uint64_t limit = wait_spin_ns();
    if (!limit) return false;
    const auto start = std::chrono::steady_clock::now();
    const auto until = start + std::chrono::nanoseconds(limit), busy = start + std::chrono::microseconds(3);
    bool yielding = false;
    for (unsigned i = 1;; ++i) {
        if (yielding) sched_yield();
        else _mm_pause();
        if (ready()) return true;
        if (yielding || i % 64 == 0) {
            const auto now = std::chrono::steady_clock::now();
            if (now > until) return false;
            yielding = now > busy;
        }
    }
}

// recvmsg that polls without blocking for a while before it blocks.
inline ssize_t recvmsg_spinning(int fd, msghdr* msg, int flags) {
    const size_t control = msg->msg_controllen;
    ssize_t n = -1;
    if (spin_until([&] {
            msg->msg_controllen = control;
            n = recvmsg(fd, msg, flags | MSG_DONTWAIT);
            return n >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK);
        }))
        return n;
    msg->msg_controllen = control;
    return recvmsg(fd, msg, flags);
}

}  // namespace cpusim
