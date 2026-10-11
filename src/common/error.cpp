#include "common/error.h"

namespace cpusim {

const char* error_name(Error e) {
    switch (e) {
        case Error::Success: return "cudaSuccess";
        case Error::InvalidValue: return "cudaErrorInvalidValue";
        case Error::MemoryAllocation: return "cudaErrorMemoryAllocation";
        case Error::InitializationError: return "cudaErrorInitializationError";
        case Error::InvalidConfiguration: return "cudaErrorInvalidConfiguration";
        case Error::InvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
        case Error::InvalidKernelImage: return "cudaErrorInvalidKernelImage";
        case Error::NoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";
        case Error::MissingConfiguration: return "cudaErrorMissingConfiguration";
        case Error::InvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
        case Error::InvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
        case Error::InvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
        case Error::InsufficientDriver: return "cudaErrorInsufficientDriver";
        case Error::DevicesUnavailable: return "cudaErrorDevicesUnavailable";
        case Error::NoDevice: return "cudaErrorNoDevice";
        case Error::InvalidDevice: return "cudaErrorInvalidDevice";
        case Error::OperatingSystem: return "cudaErrorOperatingSystem";
        case Error::IllegalAddress: return "cudaErrorIllegalAddress";
        case Error::LaunchFailure: return "cudaErrorLaunchFailure";
        case Error::NotSupported: return "cudaErrorNotSupported";
        case Error::Unknown: return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

const char* error_string(Error e) {
    switch (e) {
        case Error::Success: return "no error";
        case Error::InvalidValue: return "invalid argument";
        case Error::MemoryAllocation: return "out of memory";
        case Error::InitializationError: return "initialization error";
        case Error::InvalidConfiguration: return "invalid configuration argument";
        case Error::InvalidDeviceFunction: return "invalid device function";
        case Error::InvalidKernelImage: return "device kernel image is invalid";
        case Error::NoKernelImageForDevice: return "no kernel image is available for execution on the device";
        case Error::MissingConfiguration: return "__global__ function call is not configured";
        case Error::InvalidResourceHandle: return "invalid resource handle";
        case Error::InvalidDevicePointer: return "invalid device pointer";
        case Error::InvalidMemcpyDirection: return "invalid copy direction for memcpy";
        case Error::InsufficientDriver: return "CUDA driver version is insufficient for CUDA runtime version";
        case Error::DevicesUnavailable: return "CUDA-capable device(s) is/are busy or unavailable";
        case Error::NoDevice: return "no CUDA-capable device is detected";
        case Error::InvalidDevice: return "invalid device ordinal";
        case Error::OperatingSystem: return "OS call failed or operation not supported on this OS";
        case Error::IllegalAddress: return "an illegal memory access was encountered";
        case Error::LaunchFailure: return "unspecified launch failure";
        case Error::NotSupported: return "operation not supported";
        case Error::Unknown: return "unknown error";
    }
    return "unknown error";
}

}  // namespace cpusim
