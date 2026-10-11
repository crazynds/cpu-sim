/* CUDA runtime API (host side) implemented by the CPU SIMT runtime.
 * Written from the public documentation; enum values match CUDA 13 so that
 * programs comparing against them behave the same. */
#ifndef CPUSIM_CUDA_RUNTIME_API_H
#define CPUSIM_CUDA_RUNTIME_API_H

#include <stddef.h>

#include "vector_types.h"

#define CUDART_VERSION 13000
#define __CUDART_API_VERSION 13000

#ifdef __cplusplus
#define __CPUSIM_DEFAULT_ARG(v) = v
#else
#define __CPUSIM_DEFAULT_ARG(v)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInsufficientDriver = 35,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorMissingConfiguration = 52,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorAssert = 710,
    cudaErrorLaunchFailure = 719,
    cudaErrorNotSupported = 801,
    cudaErrorUnknown = 999
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

typedef struct CUuuid_st {
    char bytes[16];
} cudaUUID_t;

/* Subset of cudaDeviceProp; field names and meaning as in CUDA 13. */
struct cudaDeviceProp {
    char name[256];
    cudaUUID_t uuid;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int multiProcessorCount;
    int integrated;
    int canMapHostMemory;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int cooperativeLaunch;
    size_t sharedMemPerBlockOptin;
    int maxBlocksPerMultiProcessor;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int memoryPoolsSupported;
    int clockRate;
    int memoryClockRate;
};

cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaDriverGetVersion(int* version);
cudaError_t cudaRuntimeGetVersion(int* version);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
/* (cudaMemPrefetchAsync and cudaMemAdvise take a device number, as up to
 * CUDA 12; cuda_runtime.h adds the CUDA 13 forms with a cudaMemLocation.) */
/* Managed memory lives in the device's shared pool: the same address in the
 * application and in kernels, nothing ever moves (it is all host RAM), so the
 * prefetch and advice calls have nothing to do. */
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost 0x02
#define cudaMemAttachSingle 0x04
#define cudaCpuDeviceId ((int)-1)
#define cudaInvalidDeviceId ((int)-2)
enum cudaMemoryAdvise {
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6
};
enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
    cudaMemLocationTypeHost = 2,
    cudaMemLocationTypeHostNuma = 3,
    cudaMemLocationTypeHostNumaCurrent = 4
};
struct cudaMemLocation {
    enum cudaMemLocationType type;
    int id;
};
#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags __CPUSIM_DEFAULT_ARG(cudaMemAttachGlobal));
cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream __CPUSIM_DEFAULT_ARG(0));
cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);

/* Kernel launch. `func` is the host-side stub registered by the compiler. */
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem,
                             cudaStream_t stream);

/* Streams and events: each stream runs its operations in order on a thread
 * of its own, streams run concurrently, and launches and asynchronous copies
 * return right away (libcpusim_cudart, cudart/streams.h). */
#ifdef __cplusplus
#define CPUSIM_DEFAULT(v) = v
#else
#define CPUSIM_DEFAULT(v)
#endif
#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)
typedef void (*cudaHostFn_t)(void* userData);
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData);
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags CPUSIM_DEFAULT(0));
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif
