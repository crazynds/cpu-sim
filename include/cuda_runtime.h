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
template <class T>
static inline cudaError_t cudaMallocPitch(T** devPtr, size_t* pitch, size_t width, size_t height) {
    return cudaMallocPitch(reinterpret_cast<void**>(devPtr), pitch, width, height);
}
template <class T>
static inline cudaError_t cudaMallocAsync(T** devPtr, size_t size, cudaStream_t stream) {
    return cudaMallocAsync(reinterpret_cast<void**>(devPtr), size, stream);
}
/* Kernels by function: cudaFuncGetAttributes(&a, kernel). */
template <class T>
static inline cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, T* func) {
    return cudaFuncGetAttributes(attr, (const void*)func);
}
template <class T>
static inline cudaError_t cudaFuncSetAttribute(T* func, enum cudaFuncAttribute attr, int value) {
    return cudaFuncSetAttribute((const void*)func, attr, value);
}
template <class T>
static inline cudaError_t cudaFuncSetCacheConfig(T* func, enum cudaFuncCache config) {
    return cudaFuncSetCacheConfig((const void*)func, config);
}
template <class T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize,
                                                                        size_t dynamicSMemSize) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, (const void*)func, blockSize, dynamicSMemSize);
}
template <class T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, T func, int blockSize,
                                                                                 size_t dynamicSMemSize,
                                                                                 unsigned int flags) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize,
                                                                  dynamicSMemSize, flags);
}
/* The block size with the most resident threads per multiprocessor (the
 * largest among the best), and the grid that fills the device with it. */
template <class T>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
                                                             size_t dynamicSMemSize = 0, int blockSizeLimit = 0) {
    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, 0);
    if (e != cudaSuccess) return e;
    const int limit = blockSizeLimit > 0 && blockSizeLimit < p.maxThreadsPerBlock ? blockSizeLimit : p.maxThreadsPerBlock;
    int best = 0, best_threads = -1;
    for (int b = 32; b <= limit; b += 32) {
        int n = 0;
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void*)func, b, dynamicSMemSize);
        if (e != cudaSuccess) return e;
        if (n * b >= best_threads) best_threads = n * b, best = b;
    }
    if (!best) return cudaErrorInvalidValue;
    *blockSize = best;
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void*)func, best, dynamicSMemSize);
    *minGridSize = n * p.multiProcessorCount;
    return cudaSuccess;
}
template <class T>
static inline cudaError_t cudaLaunchCooperativeKernel(T* func, dim3 grid, dim3 block, void** args, size_t sharedMem = 0,
                                                      cudaStream_t stream = 0) {
    return cudaLaunchCooperativeKernel((const void*)func, grid, block, args, sharedMem, stream);
}
template <class T>
static inline cudaError_t cudaLaunchKernel(T* func, dim3 grid, dim3 block, void** args, size_t sharedMem = 0,
                                           cudaStream_t stream = 0) {
    return cudaLaunchKernel((const void*)func, grid, block, args, sharedMem, stream);
}
/* Variables by reference: cudaMemcpyToSymbol(coef, ...). */
template <class T>
static inline cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0,
                                             enum cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
    return cudaMemcpyToSymbol((const void*)&symbol, src, count, offset, kind);
}
template <class T>
static inline cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0,
                                               enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
    return cudaMemcpyFromSymbol(dst, (const void*)&symbol, count, offset, kind);
}
template <class T>
static inline cudaError_t cudaMemcpyToSymbolAsync(const T& symbol, const void* src, size_t count, size_t offset = 0,
                                                  enum cudaMemcpyKind kind = cudaMemcpyHostToDevice,
                                                  cudaStream_t stream = 0) {
    return cudaMemcpyToSymbolAsync((const void*)&symbol, src, count, offset, kind, stream);
}
template <class T>
static inline cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const T& symbol, size_t count, size_t offset = 0,
                                                    enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost,
                                                    cudaStream_t stream = 0) {
    return cudaMemcpyFromSymbolAsync(dst, (const void*)&symbol, count, offset, kind, stream);
}
template <class T>
static inline cudaError_t cudaGetSymbolAddress(void** devPtr, const T& symbol) {
    return cudaGetSymbolAddress(devPtr, (const void*)&symbol);
}
template <class T>
static inline cudaError_t cudaGetSymbolSize(size_t* size, const T& symbol) {
    return cudaGetSymbolSize(size, (const void*)&symbol);
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
