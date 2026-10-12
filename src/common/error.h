#pragma once

namespace cpusim {

// Values match cudaError_t, so the runtime passes them through unchanged.
enum class Error : int {
    Success = 0,
    InvalidValue = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    InvalidConfiguration = 9,
    InvalidSymbol = 13,
    InvalidDevicePointer = 17,
    InvalidMemcpyDirection = 21,
    InsufficientDriver = 35,
    DevicesUnavailable = 46,
    MissingConfiguration = 52,
    InvalidDeviceFunction = 98,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidKernelImage = 200,
    NoKernelImageForDevice = 209,
    UnsupportedLimit = 215,
    OperatingSystem = 304,
    InvalidResourceHandle = 400,
    IllegalAddress = 700,
    Assert = 710,
    LaunchFailure = 719,
    NotSupported = 801,
    Unknown = 999,
};

const char* error_name(Error e);
const char* error_string(Error e);

}  // namespace cpusim
