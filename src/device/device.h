#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "common/error.h"
#include "device/executor.h"
#include "device/properties.h"

namespace cpusim {

// The device interface (DESIGN.md 6.1), in the spirit of the CUDA driver API.
// Device pointers are opaque 64-bit values; with the daemon backend they are not
// valid addresses in the application.
class Device {
public:
    virtual ~Device() = default;

    virtual const char* backend_name() const = 0;
    virtual const DeviceProperties& properties() const = 0;
    virtual std::string driver_version() const = 0;

    virtual Error malloc(uint64_t bytes, uint64_t* dptr) = 0;
    virtual Error free(uint64_t dptr) = 0;
    // cudaMallocManaged: memory at the same address in the application and on
    // the device (the shared pool). NotSupported without one.
    virtual Error malloc_managed(uint64_t bytes, uint64_t* dptr) = 0;
    // True if `ptr` is in the shared pool: the application can use it directly.
    virtual bool is_managed_pointer(const void* ptr) const = 0;
    virtual Error memcpy_htod(uint64_t dst, const void* src, uint64_t n) = 0;
    virtual Error memcpy_dtoh(void* dst, uint64_t src, uint64_t n) = 0;
    virtual Error memcpy_dtod(uint64_t dst, uint64_t src, uint64_t n) = 0;
    virtual Error memset(uint64_t dst, int value, uint64_t n) = 0;
    virtual Error mem_get_info(uint64_t* free, uint64_t* total) = 0;
    virtual Error synchronize() = 0;

    // Kernel modules: `image` is the native shared object from the fatbin.
    virtual Error load_module(const void* image, size_t size, uint64_t* module) = 0;
    virtual Error get_function(uint64_t module, const std::string& name, uint64_t* function) = 0;
    // Points the module's variables at their VRAM copies: `table` has one
    // "<address> <size> <name>" line per variable. With `init`, also copies
    // each variable's initial value from this module to its address.
    virtual Error bind_symbols(uint64_t module, const std::string& table, bool init) = 0;
    virtual Error launch(uint64_t function, const LaunchConfig& cfg, const void* params, size_t param_bytes,
                         LaunchResult* result = nullptr) = 0;

    // True if `ptr` falls inside the device's address range (used to infer
    // cudaMemcpyDefault). Ranges never overlap host memory.
    virtual bool is_device_pointer(const void* ptr) const = 0;
};

// Opens the device according to $CPUSIM_BACKEND:
//   "daemon"     talk to the running driver (cpusimd)
//   "inprocess"  own the VRAM inside this process
//   unset/"auto" daemon if it is running, in-process otherwise
// On failure returns null and sets *error / *message.
std::unique_ptr<Device> open_device(Error* error, std::string* message);

std::unique_ptr<Device> open_inprocess_device(Error* error, std::string* message);
// Writes `image` into an anonymous memory file; returns the fd or -1.
int make_memfd(const char* name, const void* image, size_t size);

std::unique_ptr<Device> open_daemon_device(Error* error, std::string* message, bool quiet_if_absent);

}  // namespace cpusim
