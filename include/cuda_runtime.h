/* Umbrella header, like CUDA's: the C API, C++ conveniences and, when the file
 * is compiled as CUDA, the device-side declarations. */
#ifndef CPUSIM_CUDA_RUNTIME_H
#define CPUSIM_CUDA_RUNTIME_H

#include "cuda_runtime_api.h"

#ifdef __CUDA__
#include "__cpusim_device.h"
#endif

#ifdef __cplusplus
template <class T>
static inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(devPtr), size);
}
template <class T>
static inline cudaError_t cudaMallocManaged(T** devPtr, size_t size, unsigned int flags = 1) {
    return cudaMallocManaged(reinterpret_cast<void**>(devPtr), size, flags);
}
template <class T>
static inline cudaError_t cudaMallocHost(T** ptr, size_t size) {
    return cudaMallocHost(reinterpret_cast<void**>(ptr), size);
}
template <class T>
static inline cudaError_t cudaHostAlloc(T** ptr, size_t size, unsigned int flags) {
    return cudaHostAlloc(reinterpret_cast<void**>(ptr), size, flags);
}
template <class T>
static inline cudaError_t cudaHostGetDevicePointer(T** pDevice, void* pHost, unsigned int flags) {
    return cudaHostGetDevicePointer(reinterpret_cast<void**>(pDevice), pHost, flags);
}
/* CUDA 13 forms, with a location instead of a device number. */
static inline cudaError_t cudaMemPrefetchAsync(const void* ptr, size_t count, struct cudaMemLocation location,
                                               unsigned int /*flags*/, cudaStream_t stream = 0) {
    return cudaMemPrefetchAsync(ptr, count, location.type == cudaMemLocationTypeDevice ? location.id : cudaCpuDeviceId,
                                stream);
}
static inline cudaError_t cudaMemAdvise(const void* ptr, size_t count, enum cudaMemoryAdvise advice,
                                        struct cudaMemLocation location) {
    return cudaMemAdvise(ptr, count, advice, location.type == cudaMemLocationTypeDevice ? location.id : cudaCpuDeviceId);
}
#endif

#endif
