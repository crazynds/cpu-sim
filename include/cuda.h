#pragma once
// The CUDA driver API header. Many programs include it only for the version
// macro or out of habit; the driver API itself (cu* functions) is not
// provided, so code that calls it fails at link time.

#include "cuda_runtime.h"

#define CUDA_VERSION CUDART_VERSION
