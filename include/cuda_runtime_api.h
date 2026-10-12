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
    cudaErrorUnsupportedLimit = 215,
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

typedef enum cudaLimit {
    cudaLimitStackSize = 0,
    cudaLimitPrintfFifoSize = 1,
    cudaLimitMallocHeapSize = 2,
    cudaLimitDevRuntimeSyncDepth = 3,
    cudaLimitDevRuntimePendingLaunchCount = 4,
    cudaLimitMaxL2FetchGranularity = 5,
    cudaLimitPersistingL2CacheSize = 6
} cudaLimit;

typedef struct CUstream_st* cudaStream_t;
#define CUDART_CB  /* calling convention of host callbacks: the default */
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
enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrMemoryPoolsSupported = 115
};
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaDriverGetVersion(int* version);
cudaError_t cudaRuntimeGetVersion(int* version);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t* value, cudaLimit limit);
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

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset CPUSIM_DEFAULT(0),
                               enum cudaMemcpyKind kind CPUSIM_DEFAULT(cudaMemcpyHostToDevice));
cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset CPUSIM_DEFAULT(0),
                                 enum cudaMemcpyKind kind CPUSIM_DEFAULT(cudaMemcpyDeviceToHost));
cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset,
                                    enum cudaMemcpyKind kind, cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset,
                                      enum cudaMemcpyKind kind, cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol);
cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol);

/* Pitched (2D) and 3D memory. */
struct cudaPitchedPtr {
    void* ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};
struct cudaExtent {
    size_t width;  /* bytes for linear memory */
    size_t height;
    size_t depth;
};
struct cudaPos {
    size_t x;
    size_t y;
    size_t z;
};
typedef struct cudaArray* cudaArray_t;
struct cudaMemcpy3DParms {
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    struct cudaExtent extent;
    enum cudaMemcpyKind kind;
};
static inline struct cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) {
    struct cudaPitchedPtr r = {d, p, xsz, ysz};
    return r;
}
static inline struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
    struct cudaExtent r = {w, h, d};
    return r;
}
static inline struct cudaPos make_cudaPos(size_t x, size_t y, size_t z) {
    struct cudaPos r = {x, y, z};
    return r;
}
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                         enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                              enum cudaMemcpyKind kind, cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p);
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height);
cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height,
                              cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent,
                              cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count,
                                cudaStream_t stream CPUSIM_DEFAULT(0));
cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream);
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream);

/* Pointers. */
enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
};
struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
};
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length CPUSIM_DEFAULT(0),
                                     unsigned int flags CPUSIM_DEFAULT(cudaMemAttachSingle));
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr);
cudaError_t cudaHostGetFlags(unsigned int* flags, void* host);

/* Streams and the device: queries and settings with nothing to change here. */
cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags);
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
enum cudaFuncCache {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3
};
#define cudaDeviceScheduleAuto 0x00
#define cudaDeviceScheduleSpin 0x01
#define cudaDeviceScheduleYield 0x02
#define cudaDeviceScheduleBlockingSync 0x04
#define cudaDeviceMapHost 0x08
#define cudaDeviceLmemResizeToMax 0x10
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceFlags(unsigned int* flags);
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache config);
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* config);
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop);

/* Kernels: attributes, occupancy, cooperative launch. */
struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
};
enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9
};
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func);
cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value);
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache config);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize,
                                                          size_t dynamicSMemSize);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize,
                                                                   size_t dynamicSMemSize, unsigned int flags);
cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem,
                                        cudaStream_t stream);

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif
