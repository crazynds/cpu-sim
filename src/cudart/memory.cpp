// The CUDA runtime: device, managed, pinned and mapped allocations, copies
// and memsets (1D, pitched, 3D, peer), and memory pools.
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

extern char** environ;

#include "common/cpu_id.h"
#include "cudart/streams.h"
#include "cuda_runtime_api.h"
#include "device/device.h"
#include "device/kernel_abi.h"
#include "cudart/runtime_internal.h"

using namespace cpusim::rt;

extern "C" {

cudaError_t cudaMalloc(void** ptr, size_t size) {
    if (!ptr) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    uint64_t dptr = 0;
    Error e = d->malloc(size, &dptr);
    *ptr = e == Error::Success ? reinterpret_cast<void*>(dptr) : nullptr;
    return record(e);
}

cudaError_t cudaFree(void* ptr) {
    REQUIRE_DEVICE(d);
    runtime().sched.wait_all();  // like CUDA: no work still using it
    return record(d->free(reinterpret_cast<uint64_t>(ptr)));
}

cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int) {
    if (!ptr) return record(Error::InvalidValue);
    *ptr = nullptr;
    REQUIRE_DEVICE(d);
    uint64_t p = 0;
    Error e = d->malloc_managed(size, &p);
    *ptr = reinterpret_cast<void*>(p);
    return record(e);
}

// Nothing moves: managed memory is host RAM seen at one address by both sides.
cudaError_t cudaMemPrefetchAsync(const void*, size_t, int, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaMemAdvise(const void*, size_t, cudaMemoryAdvise, int) { return cudaSuccess; }

// Page-locked host memory comes from the shared pool: the same address in the
// application and in kernels, as page-locked memory is under unified
// addressing (a kernel may use it directly, mapped or not). Without a pool,
// ordinary memory.
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) { return cudaMallocHost(ptr, size); }


}  // extern "C"

#pragma GCC visibility push(hidden)
namespace cpusim::rt {
std::mutex mapped_mutex;
std::vector<MappedRange> mapped_ranges;
}  // namespace cpusim::rt
#pragma GCC visibility pop

extern "C" {


cudaError_t cudaHostGetDevicePointer(void** dev, void* host, unsigned int) {
    if (!dev) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    {
        std::lock_guard lock(mapped_mutex);
        for (const auto& r : mapped_ranges)
            if (static_cast<char*>(host) >= r.host && static_cast<char*>(host) < r.host + r.size) {
                *dev = r.shadow + (static_cast<char*>(host) - r.host);
                return cudaSuccess;
            }
    }
    if (!d->is_managed_pointer(host)) return record(Error::InvalidValue);  // not from cudaHostAlloc
    *dev = host;
    return cudaSuccess;
}

// Page locking means nothing here. Mapped registration (kernels use the
// memory in place): in-process, the program's memory is the kernels'; with
// the driver, kernels run in its process and see a copy in the shared pool,
// made equal to the program's memory around every launch (copied in before,
// back after), so a kernel reads what the program wrote before launching it
// and the program reads what the kernel wrote once it is done.
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    if (!ptr || !size) return record(Error::InvalidValue);
    if (flags & cudaHostRegisterMapped) {
        REQUIRE_DEVICE(d);
        char* shadow = static_cast<char*>(ptr);
        if (std::strcmp(d->backend_name(), "inprocess") != 0) {
            uint64_t p = 0;
            if (Error e = d->malloc_managed(size, &p); e != Error::Success) return record(e);
            shadow = reinterpret_cast<char*>(p);
            std::memcpy(shadow, ptr, size);
        }
        std::lock_guard lock(mapped_mutex);
        mapped_ranges.push_back({static_cast<char*>(ptr), shadow, size});
    }
    cpusim::cudart::pinned_add(ptr, size, false);
    return cudaSuccess;
}
cudaError_t cudaHostUnregister(void* ptr) {
    {
        std::lock_guard lock(mapped_mutex);
        for (auto it = mapped_ranges.begin(); it != mapped_ranges.end(); ++it)
            if (it->host == ptr) {
                if (it->shadow != it->host) {
                    runtime().sched.wait_all();
                    std::memcpy(it->host, it->shadow, it->size);
                    cudaFree(it->shadow);
                }
                mapped_ranges.erase(it);
                break;
            }
    }
    return cpusim::cudart::pinned_remove(ptr) ? cudaSuccess : record(Error::InvalidValue);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    if (!ptr) return record(Error::InvalidValue);
    *ptr = nullptr;
    if (!size) return cudaSuccess;
    REQUIRE_DEVICE(d);
    uint64_t p = 0;
    if (Error e = d->malloc_managed(size, &p); e == Error::Success) {
        *ptr = reinterpret_cast<void*>(p);
        cpusim::cudart::pinned_add(*ptr, size, false);  // page-locked host memory, for the queries
    } else if (e == Error::NotSupported) {
        // Recorded so asynchronous copies from it are not staged.
        *ptr = std::malloc(size);
        if (!*ptr) return record(Error::MemoryAllocation);
        cpusim::cudart::pinned_add(*ptr, size, true);
    } else {
        return record(e);
    }
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    if (!ptr) return cudaSuccess;
    cudaError_t init_err = cudaSuccess;
    if (cpusim::Device* d = device(&init_err); d && d->is_managed_pointer(ptr)) {
        cpusim::cudart::pinned_remove(ptr);
        return cudaFree(ptr);
    }
    // Only what cudaMallocHost allocated itself (not a registered buffer).
    if (!cpusim::cudart::pinned_owned(ptr)) return record(Error::InvalidValue);
    runtime().sched.wait_all();  // no copy still reading it
    cpusim::cudart::pinned_remove(ptr);
    std::free(ptr);
    return cudaSuccess;
}


}  // extern "C"

#pragma GCC visibility push(hidden)
namespace cpusim::rt {

// Which way a copy goes, with managed memory counted as host memory (it is,
// on this side).
cudaMemcpyKind resolve_kind(cpusim::Device* d, void* dst, const void* src, cudaMemcpyKind kind) {
    bool dm = d->is_managed_pointer(dst), sm = d->is_managed_pointer(src);
    bool dd = d->is_device_pointer(dst), sd = d->is_device_pointer(src);
    if (dm || sm || kind == cudaMemcpyDefault)
        return dd ? (sd ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                  : (sd ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    return kind;
}

cudaError_t copy_now(cpusim::Device* d, void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    auto D = [](const void* p) { return reinterpret_cast<uint64_t>(p); };
    switch (kind) {
        case cudaMemcpyHostToHost: std::memmove(dst, src, count); return cudaSuccess;
        case cudaMemcpyHostToDevice: return static_cast<cudaError_t>(d->memcpy_htod(D(dst), src, count));
        case cudaMemcpyDeviceToHost: return static_cast<cudaError_t>(d->memcpy_dtoh(dst, D(src), count));
        case cudaMemcpyDeviceToDevice: return static_cast<cudaError_t>(d->memcpy_dtod(D(dst), D(src), count));
        default: return cudaErrorInvalidMemcpyDirection;
    }
}

// Queues a copy on `stream`. Host data an asynchronous copy reads is staged
// when the copy cannot run right away (CUDA's rule for pageable memory: the
// buffer may be reused as soon as the call returns), unless it is pinned or
// managed memory.
cpusim::cudart::CompletionPtr queue_copy(cpusim::Device* d, void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                         cudaStream_t stream, bool sync) {
    kind = resolve_kind(d, dst, src, kind);
    auto staged = std::make_shared<std::vector<uint8_t>>();
    const bool host_src = kind == cudaMemcpyHostToDevice || kind == cudaMemcpyHostToHost;
    const bool stage = host_src && !sync && !d->is_managed_pointer(src) && !cpusim::cudart::pinned_contains(src, count);
    cpusim::cudart::EnqueueOptions opt;
    opt.inline_if_idle = true;
    opt.sync = sync;
    if (stage) opt.on_queue = [staged, src, count] { staged->assign(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + count); };
    return runtime().sched.enqueue(
        stream,
        [d, dst, src, count, kind, staged] { return copy_now(d, dst, staged->empty() ? src : staged->data(), count, kind); },
        std::move(opt));
}

cpusim::cudart::CompletionPtr queue_memset(cpusim::Device* d, void* ptr, int value, size_t count, cudaStream_t stream) {
    return runtime().sched.enqueue(stream, [d, ptr, value, count]() -> cudaError_t {
        if (d->is_managed_pointer(ptr)) {
            std::memset(ptr, value, count);
            return cudaSuccess;
        }
        return static_cast<cudaError_t>(d->memset(reinterpret_cast<uint64_t>(ptr), value, count));
    });
}

}  // namespace cpusim::rt
#pragma GCC visibility pop

extern "C" {

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    // Synchronous: in order on the default stream, and back when done.
    auto done = queue_copy(d, dst, src, count, kind, nullptr, true);
    done->wait();
    if (done->error != cudaSuccess) return record(done->error);
    return async_error();
}

cudaError_t cudaMemset(void* ptr, int value, size_t count) {
    if (count == 0) return cudaSuccess;
    REQUIRE_DEVICE(d);
    if (!d->is_device_pointer(ptr) && !d->is_managed_pointer(ptr)) return record(Error::InvalidValue);
    auto done = queue_memset(d, ptr, value, count, nullptr);  // asynchronous on device memory, like CUDA's
    // On managed memory the host may look right after (the GPU's memset is
    // done by then too): wait.
    if (d->is_managed_pointer(ptr)) {
        done->wait();
        return async_error();
    }
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    if (!free || !total) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    uint64_t f = 0, t = 0;
    Error e = d->mem_get_info(&f, &t);
    *free = f;
    *total = t;
    return record(e);
}


}  // extern "C"

extern "C" {

// ---- pitched and 3D memory

cudaError_t cudaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
    if (!ptr || !pitch) return record(Error::InvalidValue);
    // Rows aligned as a GPU aligns them (512 bytes): pitch arithmetic in
    // programs works the same.
    const size_t p = (std::max<size_t>(width, 1) + 511) / 512 * 512;
    cudaError_t e = cudaMalloc(ptr, p * std::max<size_t>(height, 1));
    if (e == cudaSuccess) *pitch = p;
    return e;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* out, struct cudaExtent extent) {
    if (!out) return record(Error::InvalidValue);
    size_t pitch = 0;
    void* p = nullptr;
    cudaError_t e = cudaMallocPitch(&p, &pitch, extent.width, extent.height * std::max<size_t>(extent.depth, 1));
    if (e == cudaSuccess) *out = make_cudaPitchedPtr(p, pitch, extent.width, extent.height);
    return e;
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                              enum cudaMemcpyKind kind, cudaStream_t stream) {
    if (width > dpitch || width > spitch) return record(Error::InvalidValue);
    if (!width || !height) return cudaSuccess;
    // Contiguous rows: one copy.
    if (dpitch == width && spitch == width)
        return cudaMemcpyAsync(dst, src, width * height, kind, stream);
    for (size_t y = 0; y < height; ++y)
        if (cudaError_t e = cudaMemcpyAsync(static_cast<char*>(dst) + y * dpitch,
                                            static_cast<const char*>(src) + y * spitch, width, kind, stream);
            e != cudaSuccess)
            return e;
    return cudaSuccess;
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                         enum cudaMemcpyKind kind) {
    if (cudaError_t e = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, nullptr); e != cudaSuccess)
        return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream) {
    if (!p || p->srcArray || p->dstArray) return record(p ? Error::NotSupported : Error::InvalidValue);
    const cudaPitchedPtr& s = p->srcPtr;
    const cudaPitchedPtr& t = p->dstPtr;
    const size_t sh = s.ysize ? s.ysize : p->extent.height, th = t.ysize ? t.ysize : p->extent.height;
    for (size_t z = 0; z < p->extent.depth; ++z) {
        const char* from = static_cast<const char*>(s.ptr) + ((p->srcPos.z + z) * sh + p->srcPos.y) * s.pitch + p->srcPos.x;
        char* to = static_cast<char*>(t.ptr) + ((p->dstPos.z + z) * th + p->dstPos.y) * t.pitch + p->dstPos.x;
        if (cudaError_t e = cudaMemcpy2DAsync(to, t.pitch, from, s.pitch, p->extent.width, p->extent.height, p->kind,
                                              stream);
            e != cudaSuccess)
            return e;
    }
    return cudaSuccess;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p) {
    if (cudaError_t e = cudaMemcpy3DAsync(p, nullptr); e != cudaSuccess) return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemset2DAsync(void* ptr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
    if (width > pitch) return record(Error::InvalidValue);
    if (pitch == width) return cudaMemsetAsync(ptr, value, width * height, stream);
    for (size_t y = 0; y < height; ++y)
        if (cudaError_t e = cudaMemsetAsync(static_cast<char*>(ptr) + y * pitch, value, width, stream); e != cudaSuccess)
            return e;
    return cudaSuccess;
}

cudaError_t cudaMemset2D(void* ptr, size_t pitch, int value, size_t width, size_t height) {
    if (cudaError_t e = cudaMemset2DAsync(ptr, pitch, value, width, height, nullptr); e != cudaSuccess) return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr p, int value, struct cudaExtent extent, cudaStream_t stream) {
    const size_t rows = p.ysize ? p.ysize : extent.height;
    for (size_t z = 0; z < extent.depth; ++z)
        if (cudaError_t e = cudaMemset2DAsync(static_cast<char*>(p.ptr) + z * rows * p.pitch, p.pitch, value,
                                              extent.width, extent.height, stream);
            e != cudaSuccess)
            return e;
    return cudaSuccess;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr p, int value, struct cudaExtent extent) {
    if (cudaError_t e = cudaMemset3DAsync(p, value, extent, nullptr); e != cudaSuccess) return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dst_device, const void* src, int src_device, size_t count,
                                cudaStream_t stream) {
    if (dst_device != 0 || src_device != 0) return record(Error::InvalidDevice);
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

cudaError_t cudaMemcpyPeer(void* dst, int dst_device, const void* src, int src_device, size_t count) {
    if (cudaError_t e = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, nullptr); e != cudaSuccess)
        return e;
    return cudaStreamSynchronize(nullptr);
}

// Stream-ordered allocation: the allocation itself is immediate; the free
// waits for the stream's earlier work.
cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t) { return cudaMalloc(ptr, size); }
cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
    if (cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) return e;
    return cudaFree(ptr);
}

// Memory pools: the device's allocator is its one pool (a handle to tell it
// apart from null); attributes set are kept, the amounts are the device's.
namespace {
struct MemPool {
    uint64_t threshold = 0, reuse[3] = {1, 1, 1};
    uint64_t used_high = 0;
};
MemPool& default_pool() {
    static MemPool p;
    return p;
}
}  // namespace

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) {
    if (!memPool) return record(Error::InvalidValue);
    if (device != 0) return record(Error::InvalidDevice);
    *memPool = reinterpret_cast<cudaMemPool_t>(&default_pool());
    return cudaSuccess;
}
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) { return cudaDeviceGetDefaultMemPool(memPool, device); }
cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
    if (device != 0) return record(Error::InvalidDevice);
    return memPool == reinterpret_cast<cudaMemPool_t>(&default_pool()) ? cudaSuccess : record(Error::InvalidValue);
}
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value) {
    auto* p = reinterpret_cast<MemPool*>(memPool);
    if (p != &default_pool() || !value) return record(Error::InvalidValue);
    switch (attr) {
        case cudaMemPoolAttrReleaseThreshold: p->threshold = *static_cast<uint64_t*>(value); return cudaSuccess;
        case cudaMemPoolReuseFollowEventDependencies:
        case cudaMemPoolReuseAllowOpportunistic:
        case cudaMemPoolReuseAllowInternalDependencies:
            p->reuse[attr - 1] = *static_cast<int*>(value);
            return cudaSuccess;
        case cudaMemPoolAttrReservedMemHigh:
        case cudaMemPoolAttrUsedMemHigh:
            p->used_high = 0;  // only resetting to 0 is allowed
            return cudaSuccess;
        default: return record(Error::InvalidValue);
    }
}
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value) {
    auto* p = reinterpret_cast<MemPool*>(memPool);
    if (p != &default_pool() || !value) return record(Error::InvalidValue);
    size_t free = 0, total = 0;
    switch (attr) {
        case cudaMemPoolAttrReleaseThreshold: *static_cast<uint64_t*>(value) = p->threshold; return cudaSuccess;
        case cudaMemPoolReuseFollowEventDependencies:
        case cudaMemPoolReuseAllowOpportunistic:
        case cudaMemPoolReuseAllowInternalDependencies:
            *static_cast<int*>(value) = int(p->reuse[attr - 1]);
            return cudaSuccess;
        default:
            if (cudaError_t e = cudaMemGetInfo(&free, &total); e != cudaSuccess) return e;
            p->used_high = std::max<uint64_t>(p->used_high, total - free);
            *static_cast<uint64_t*>(value) =
                attr == cudaMemPoolAttrUsedMemHigh || attr == cudaMemPoolAttrReservedMemHigh ? p->used_high : total - free;
            return cudaSuccess;
    }
}
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t) {
    return reinterpret_cast<MemPool*>(memPool) == &default_pool() ? cudaSuccess : record(Error::InvalidValue);
}
cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) {
    if (reinterpret_cast<MemPool*>(memPool) != &default_pool()) return record(Error::InvalidValue);
    return cudaMallocAsync(ptr, size, stream);
}

// Peer access: one device, so no peer to reach.
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    if (!canAccessPeer) return record(Error::InvalidValue);
    if (device != 0 || peerDevice != 0) return record(Error::InvalidDevice);
    *canAccessPeer = 0;
    return cudaSuccess;
}
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int) {
    return record(peerDevice == 0 ? Error::InvalidValue : Error::InvalidDevice);
}
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    return record(peerDevice == 0 ? Error::InvalidValue : Error::InvalidDevice);
}


}  // extern "C"
