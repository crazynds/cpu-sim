// The CUDA runtime API on top of the device interface. One device (0); the
// backend is chosen on first use (see open_device). The device, streams,
// events, errors and settings; the other parts are listed in runtime_internal.h.
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

#pragma GCC visibility push(hidden)
namespace cpusim::rt {

Runtime& runtime() {
    static Runtime rt;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string message;
        rt.device = cpusim::open_device(&rt.init_error, &message);
        if (!rt.device && std::getenv("CPUSIM_QUIET") == nullptr)
            std::fprintf(stderr, "cpusim: no device: %s\n", message.c_str());
    });
    return rt;
}

thread_local cudaError_t last_error = cudaSuccess;
// Threads per warp the program's kernels were compiled for (CPUSIM_WARP_SIZE
// at compile time, in each module's manifest): what warpSize reports.
std::atomic<int> program_warp_size{0};
int warp_size() {
    int w = program_warp_size.load();
    return w ? w : CPUSIM_WARP_SIZE;
}

cudaError_t record(Error e) {
    auto err = static_cast<cudaError_t>(e);
    if (err != cudaSuccess) last_error = err;
    return err;
}
cudaError_t record(cudaError_t err) {
    if (err != cudaSuccess) last_error = err;
    return err;
}

// At a synchronization point: the first error of asynchronous work, if any.
cudaError_t async_error() { return record(runtime().sched.take_error()); }

// Returns the device or records why there is none.
cpusim::Device* device(cudaError_t* err) {
    Runtime& rt = runtime();
    if (!rt.device) *err = record(rt.init_error == Error::Success ? Error::NoDevice : rt.init_error);
    return rt.device.get();
}

}  // namespace cpusim::rt
#pragma GCC visibility pop

extern "C" {

cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return record(Error::InvalidValue);
    cudaError_t err = cudaSuccess;
    *count = device(&err) ? 1 : 0;
    return err;
}

cudaError_t cudaGetDevice(int* dev) {
    if (!dev) return record(Error::InvalidValue);
    *dev = 0;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int dev) {
    REQUIRE_DEVICE(d);
    (void)d;
    return dev == 0 ? cudaSuccess : record(Error::InvalidDevice);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int dev) {
    if (!prop) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    if (dev != 0) return record(Error::InvalidDevice);
    const auto& p = d->properties();
    *prop = {};
    std::memcpy(prop->name, p.name, sizeof(prop->name));
    std::memcpy(prop->uuid.bytes, p.uuid, sizeof(prop->uuid.bytes));
#define COPY(f) prop->f = p.f
#define COPY3(f) COPY(f[0]), COPY(f[1]), COPY(f[2])
    COPY(totalGlobalMem); COPY(sharedMemPerBlock); COPY(regsPerBlock); COPY(memPitch);
    prop->warpSize = warp_size();
    COPY(maxThreadsPerBlock); COPY3(maxThreadsDim); COPY3(maxGridSize); COPY(totalConstMem); COPY(major);
    COPY(minor); COPY(textureAlignment); COPY(texturePitchAlignment); COPY(multiProcessorCount); COPY(integrated);
    COPY(canMapHostMemory); COPY(concurrentKernels); COPY(ECCEnabled); COPY(pciBusID); COPY(pciDeviceID);
    COPY(pciDomainID); COPY(asyncEngineCount); COPY(unifiedAddressing); COPY(memoryBusWidth); COPY(l2CacheSize);
    COPY(persistingL2CacheMaxSize); COPY(maxThreadsPerMultiProcessor); COPY(streamPrioritiesSupported);
    COPY(globalL1CacheSupported); COPY(localL1CacheSupported); COPY(sharedMemPerMultiprocessor);
    COPY(regsPerMultiprocessor); COPY(managedMemory); COPY(isMultiGpuBoard); COPY(pageableMemoryAccess);
    COPY(concurrentManagedAccess); COPY(computePreemptionSupported); COPY(cooperativeLaunch);
    COPY(sharedMemPerBlockOptin); COPY(maxBlocksPerMultiProcessor); COPY(reservedSharedMemPerBlock);
    COPY(hostRegisterSupported); COPY(memoryPoolsSupported); COPY(clockRate); COPY(memoryClockRate);
#undef COPY3
#undef COPY
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int* version) {
    if (!version) return record(Error::InvalidValue);
    *version = runtime().device ? CUDART_VERSION : 0;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int* version) {
    if (!version) return record(Error::InvalidValue);
    *version = CUDART_VERSION;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device) {
    if (!value) return record(Error::InvalidValue);
    cudaDeviceProp p;
    if (cudaError_t e = cudaGetDeviceProperties(&p, device); e != cudaSuccess) return e;
    switch (attr) {
        case cudaDevAttrMaxThreadsPerBlock: *value = int(p.maxThreadsPerBlock); break;
        case cudaDevAttrMaxBlockDimX: *value = int(p.maxThreadsDim[0]); break;
        case cudaDevAttrMaxBlockDimY: *value = int(p.maxThreadsDim[1]); break;
        case cudaDevAttrMaxBlockDimZ: *value = int(p.maxThreadsDim[2]); break;
        case cudaDevAttrMaxGridDimX: *value = int(p.maxGridSize[0]); break;
        case cudaDevAttrMaxGridDimY: *value = int(p.maxGridSize[1]); break;
        case cudaDevAttrMaxGridDimZ: *value = int(p.maxGridSize[2]); break;
        case cudaDevAttrMaxSharedMemoryPerBlock: *value = int(p.sharedMemPerBlock); break;
        case cudaDevAttrTotalConstantMemory: *value = int(p.totalConstMem); break;
        case cudaDevAttrWarpSize: *value = warp_size(); break;
        case cudaDevAttrMaxPitch: *value = int(p.memPitch); break;
        case cudaDevAttrMaxRegistersPerBlock: *value = int(p.regsPerBlock); break;
        case cudaDevAttrClockRate: *value = int(p.clockRate); break;
        case cudaDevAttrTextureAlignment: *value = int(p.textureAlignment); break;
        case cudaDevAttrGpuOverlap: *value = int(p.asyncEngineCount > 0); break;
        case cudaDevAttrMultiProcessorCount: *value = int(p.multiProcessorCount); break;
        case cudaDevAttrKernelExecTimeout: *value = 0; break;
        case cudaDevAttrIntegrated: *value = int(p.integrated); break;
        case cudaDevAttrCanMapHostMemory: *value = int(p.canMapHostMemory); break;
        case cudaDevAttrComputeMode: *value = 0; break;
        case cudaDevAttrConcurrentKernels: *value = int(p.concurrentKernels); break;
        case cudaDevAttrEccEnabled: *value = int(p.ECCEnabled); break;
        case cudaDevAttrPciBusId: *value = int(p.pciBusID); break;
        case cudaDevAttrPciDeviceId: *value = int(p.pciDeviceID); break;
        case cudaDevAttrTccDriver: *value = 0; break;
        case cudaDevAttrMemoryClockRate: *value = int(p.memoryClockRate); break;
        case cudaDevAttrGlobalMemoryBusWidth: *value = int(p.memoryBusWidth); break;
        case cudaDevAttrL2CacheSize: *value = int(p.l2CacheSize); break;
        case cudaDevAttrMaxThreadsPerMultiProcessor: *value = int(p.maxThreadsPerMultiProcessor); break;
        case cudaDevAttrAsyncEngineCount: *value = int(p.asyncEngineCount); break;
        case cudaDevAttrUnifiedAddressing: *value = int(p.unifiedAddressing); break;
        case cudaDevAttrPciDomainId: *value = int(p.pciDomainID); break;
        case cudaDevAttrTexturePitchAlignment: *value = int(p.texturePitchAlignment); break;
        case cudaDevAttrComputeCapabilityMajor: *value = int(p.major); break;
        case cudaDevAttrComputeCapabilityMinor: *value = int(p.minor); break;
        case cudaDevAttrStreamPrioritiesSupported: *value = int(p.streamPrioritiesSupported); break;
        case cudaDevAttrGlobalL1CacheSupported: *value = int(p.globalL1CacheSupported); break;
        case cudaDevAttrLocalL1CacheSupported: *value = int(p.localL1CacheSupported); break;
        case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *value = int(p.sharedMemPerMultiprocessor); break;
        case cudaDevAttrMaxRegistersPerMultiprocessor: *value = int(p.regsPerMultiprocessor); break;
        case cudaDevAttrManagedMemory: *value = int(p.managedMemory); break;
        case cudaDevAttrIsMultiGpuBoard: *value = int(p.isMultiGpuBoard); break;
        case cudaDevAttrMultiGpuBoardGroupID: *value = 0; break;
        case cudaDevAttrPageableMemoryAccess: *value = int(p.pageableMemoryAccess); break;
        case cudaDevAttrConcurrentManagedAccess: *value = int(p.concurrentManagedAccess); break;
        case cudaDevAttrComputePreemptionSupported: *value = int(p.computePreemptionSupported); break;
        case cudaDevAttrCanUseHostPointerForRegisteredMem: *value = 1; break;
        case cudaDevAttrCooperativeLaunch: *value = int(p.cooperativeLaunch); break;
        case cudaDevAttrCooperativeMultiDeviceLaunch: *value = 0; break;
        case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = int(p.sharedMemPerBlockOptin); break;
        case cudaDevAttrCanFlushRemoteWrites: *value = 0; break;
        case cudaDevAttrHostRegisterSupported: *value = int(p.hostRegisterSupported); break;
        case cudaDevAttrPageableMemoryAccessUsesHostPageTables: *value = int(p.pageableMemoryAccess); break;
        case cudaDevAttrDirectManagedMemAccessFromHost: *value = 1; break;
        case cudaDevAttrMaxBlocksPerMultiprocessor: *value = int(p.maxBlocksPerMultiProcessor); break;
        case cudaDevAttrMaxPersistingL2CacheSize: *value = int(p.persistingL2CacheMaxSize); break;
        case cudaDevAttrMaxAccessPolicyWindowSize: *value = 0; break;
        case cudaDevAttrReservedSharedMemoryPerBlock: *value = int(p.reservedSharedMemPerBlock); break;
        case cudaDevAttrMemoryPoolsSupported: *value = int(p.memoryPoolsSupported); break;
        default: return record(Error::InvalidValue);
    }
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    REQUIRE_DEVICE(d);
    runtime().sched.wait_all();
    if (Error e = d->synchronize(); e != Error::Success) return record(e);
    return async_error();
}

// Limits are recorded and reported back; none constrains the device. Device
// code runs on the workers' own stacks (megabytes, not CUDA's kilobyte), and
// there is no device printf FIFO or heap to size.
namespace {
std::mutex limits_mutex;
size_t limits[7] = {1024, 1 << 20, 8 << 20, 2, 2048, 32, 0};
}  // namespace

cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
    if (unsigned(limit) >= 7) return record(Error::UnsupportedLimit);
    std::lock_guard lock(limits_mutex);
    limits[limit] = value;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t* value, cudaLimit limit) {
    if (!value) return record(Error::InvalidValue);
    if (unsigned(limit) >= 7) return record(Error::UnsupportedLimit);
    std::lock_guard lock(limits_mutex);
    *value = limits[limit];
    return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    cudaDeviceSynchronize();
    runtime().sched.clear_error();
    last_error = cudaSuccess;
    return cudaSuccess;
}


}  // extern "C"

extern "C" {

cudaError_t cudaStreamCreate(cudaStream_t* stream) { return cudaStreamCreateWithFlags(stream, cudaStreamDefault); }

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    return cudaStreamCreateWithPriority(stream, flags, 0);
}

// Priorities are recorded and reported; streams are not scheduled by them.
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority) {
    if (!stream) return record(Error::InvalidValue);
    *stream = new CUstream_st{runtime().sched.create(flags), flags, std::clamp(priority, -5, 0)};
    return cudaSuccess;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
    if (least) *least = 0;
    if (greatest) *greatest = -5;
    return cudaSuccess;
}


cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!is_object(stream)) return record(Error::InvalidResourceHandle);
    runtime().sched.destroy(stream->queue);  // its work still runs first
    delete stream;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    runtime().sched.wait_stream(stream);
    return async_error();
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags) {
    if (!flags) return record(Error::InvalidValue);
    *flags = is_object(stream) ? stream->flags : cudaStreamDefault;
    return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    if (!priority) return record(Error::InvalidValue);
    *priority = is_object(stream) ? stream->priority : 0;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return runtime().sched.idle(stream) ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int) {
    if (!event) return record(Error::InvalidResourceHandle);
    {
        // An event recorded during a capture: this stream joins the capture
        // (a fork); its operations go into the same graph.
        std::lock_guard lock(event->mutex);
        if (event->captured) {
            std::lock_guard clock(capture_mutex);
            capturing_streams[stream] = event->captured;
            return cudaSuccess;
        }
    }
    if (capturing(stream)) return cudaSuccess;  // ordering inside a graph is its node order
    cpusim::cudart::EnqueueOptions opt;
    {
        std::lock_guard lock(event->mutex);
        if (event->pending) opt.after.push_back(event->pending);
    }
    runtime().sched.enqueue(stream, nullptr, std::move(opt));
    return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* data) {
    if (!fn) return record(Error::InvalidValue);
    if (CUgraph_st* g = capturing(stream)) {
        CUgraphNode_st* n = add_node(g, CUgraphNode_st::Host);
        n->fn = fn, n->data = data;
        return cudaSuccess;
    }
    runtime().sched.enqueue(stream, [fn, data] {
        fn(data);
        return cudaSuccess;
    });
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t cb, void* data, unsigned int) {
    if (!cb) return record(Error::InvalidValue);
    runtime().sched.enqueue(stream, [stream, cb, data] {
        cb(stream, cudaSuccess, data);
        return cudaSuccess;
    });
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return record(Error::InvalidValue);
    if (CUgraph_st* g = capturing(stream)) {
        CUgraphNode_st* n = add_node(g, CUgraphNode_st::Memcpy);
        n->dst = dst, n->src = src, n->count = count, n->copy_kind = kind;
        return cudaSuccess;
    }
    REQUIRE_DEVICE(d);
    queue_copy(d, dst, src, count, kind, stream, false);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (CUgraph_st* g = capturing(stream)) {
        set_memset(add_node(g, CUgraphNode_st::Memset), {ptr, count, unsigned(value & 0xff), 1, count, 1});
        return cudaSuccess;
    }
    REQUIRE_DEVICE(d);
    if (!d->is_device_pointer(ptr) && !d->is_managed_pointer(ptr)) return record(Error::InvalidValue);
    queue_memset(d, ptr, value, count, stream);
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) { return cudaEventCreateWithFlags(event, cudaEventDefault); }

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (!event) return record(Error::InvalidValue);
    *event = new CUevent_st;
    (*event)->flags = flags;
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;  // a record still queued keeps its own time stamp alive
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return record(Error::InvalidResourceHandle);
    if (CUgraph_st* g = capturing(stream)) {
        std::lock_guard lock(event->mutex);
        event->captured = g;
        return cudaSuccess;
    }
    auto when = std::make_shared<std::chrono::steady_clock::time_point>();
    auto done = runtime().sched.enqueue(stream, [when] {
        *when = std::chrono::steady_clock::now();  // everything before it in the stream ran
        return cudaSuccess;
    });
    std::lock_guard lock(event->mutex);
    event->pending = done;
    event->when = when;
    event->captured = nullptr;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) return record(Error::InvalidResourceHandle);
    cpusim::cudart::CompletionPtr c;
    {
        std::lock_guard lock(event->mutex);
        c = event->pending;
    }
    if (c) c->wait();
    return async_error();
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!event) return record(Error::InvalidResourceHandle);
    std::lock_guard lock(event->mutex);
    return !event->pending || event->pending->ready() ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) return record(Error::InvalidValue);
    if (!start || !end) return record(Error::InvalidResourceHandle);
    std::scoped_lock lock(start->mutex, end->mutex);
    if (!start->pending || !end->pending || (start->flags | end->flags) & cudaEventDisableTiming)
        return record(Error::InvalidResourceHandle);
    if (!start->pending->ready() || !end->pending->ready()) return cudaErrorNotReady;
    *ms = std::chrono::duration<float, std::milli>(*end->when - *start->when).count();
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) {
    cudaError_t e = last_error;
    last_error = cudaSuccess;
    if (e == cudaSuccess) e = runtime().sched.take_error();  // from asynchronous work
    return e;
}

cudaError_t cudaPeekAtLastError(void) { return last_error != cudaSuccess ? last_error : runtime().sched.peek_error(); }

}  // extern "C"

extern "C" {

// ---- pointers, flags and settings

// Managed memory is visible everywhere at all times: nothing to attach.
cudaError_t cudaStreamAttachMemAsync(cudaStream_t, void*, size_t, unsigned int) { return cudaSuccess; }

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* a, const void* ptr) {
    if (!a) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    *a = {};
    a->device = 0;
    if (cpusim::cudart::pinned_contains(ptr, 1)) {  // page-locked: usable by kernels in place
        a->type = cudaMemoryTypeHost;
        a->hostPointer = const_cast<void*>(ptr);
        a->devicePointer = d->is_managed_pointer(ptr) ? const_cast<void*>(ptr) : nullptr;
    } else if (d->is_managed_pointer(ptr)) {
        a->type = cudaMemoryTypeManaged;
        a->devicePointer = a->hostPointer = const_cast<void*>(ptr);
    } else if (d->is_device_pointer(ptr)) {
        a->type = cudaMemoryTypeDevice;
        a->devicePointer = const_cast<void*>(ptr);
    } else {
        a->type = cudaMemoryTypeUnregistered;
        a->device = -2;
    }
    return cudaSuccess;
}

cudaError_t cudaHostGetFlags(unsigned int* flags, void* host) {
    if (!flags) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    if (d->is_managed_pointer(host)) *flags = cudaHostAllocMapped;
    else if (cpusim::cudart::pinned_contains(host, 1)) *flags = cudaHostAllocDefault;
    else return record(Error::InvalidValue);
    return cudaSuccess;
}

namespace {
unsigned device_flags = 0;
cudaFuncCache cache_config = cudaFuncCachePreferNone;
}  // namespace

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    device_flags = flags;
    return cudaSuccess;
}
cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    if (!flags) return record(Error::InvalidValue);
    *flags = device_flags | cudaDeviceMapHost;  // host memory is always mapped
    return cudaSuccess;
}
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache config) {
    cache_config = config;
    return cudaSuccess;
}
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* config) {
    if (!config) return record(Error::InvalidValue);
    *config = cache_config;
    return cudaSuccess;
}
cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }
cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp*) {
    if (!device) return record(Error::InvalidValue);
    *device = 0;
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) { return cpusim::error_string(static_cast<Error>(error)); }
const char* cudaGetErrorName(cudaError_t error) { return cpusim::error_name(static_cast<Error>(error)); }

}  // extern "C"
