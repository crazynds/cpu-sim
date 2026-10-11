/* Half precision (IEEE binary16) for CUDA source on the CPU: __half,
 * __half2 and CUDA's functions on them, usable in host and device code.
 * x86 without AVX512-FP16 has no half arithmetic: values convert to float
 * with F16C (exact) and every operation rounds back once, which gives the
 * same correctly rounded results as the GPU (see __cpusim_fp16_common.h). */
#ifndef CPUSIM_CUDA_FP16_H
#define CPUSIM_CUDA_FP16_H
#define __CUDA_FP16_H__
#define __CUDA_FP16_TYPES_EXIST__

#include "__cpusim_fp16_common.h"

#define __CPUSIM_T __half
#define __CPUSIM_T2 __half2
#define __CPUSIM_RAW __half_raw
#define __CPUSIM_RAW2 __half2_raw
#define __CPUSIM_N half
#define __CPUSIM_N2 half2
#define __CPUSIM_TOF(bits) __cpusim_fp::half_tof(bits)
#define __CPUSIM_FROM(f, mode) __cpusim_fp::half_from((f), (mode))
#define __CPUSIM_INF_BITS 0x7c00u
#include "__cpusim_fp16_impl.h"
#undef __CPUSIM_T
#undef __CPUSIM_T2
#undef __CPUSIM_RAW
#undef __CPUSIM_RAW2
#undef __CPUSIM_N
#undef __CPUSIM_N2
#undef __CPUSIM_TOF
#undef __CPUSIM_FROM
#undef __CPUSIM_INF_BITS

typedef __half half;
typedef __half2 half2;
typedef __half __nv_half;
typedef __half2 __nv_half2;

#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7c00u)
#define CUDART_NAN_FP16 __ushort_as_half((unsigned short)0x7fffu)
#define CUDART_MIN_DENORM_FP16 __ushort_as_half((unsigned short)0x0001u)
#define CUDART_MAX_NORMAL_FP16 __ushort_as_half((unsigned short)0x7bffu)
#define CUDART_NEG_ZERO_FP16 __ushort_as_half((unsigned short)0x8000u)
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000u)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3c00u)

#endif
