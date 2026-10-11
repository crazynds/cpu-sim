/* bfloat16 for CUDA source on the CPU: __nv_bfloat16, __nv_bfloat162 and
 * CUDA's functions on them, usable in host and device code. A bfloat16 is the
 * high half of a float: widening is a shift, narrowing rounds to nearest even
 * on the bits, and arithmetic is done in float and rounded once (correctly
 * rounded, see __cpusim_fp16_common.h). The dot products of the tensor core
 * emulation use AVX512-BF16 when the CPU has it (mma.h). */
#ifndef CPUSIM_CUDA_BF16_H
#define CPUSIM_CUDA_BF16_H
#define __CUDA_BF16_H__
#define __CUDA_BF16_TYPES_EXIST__

#include "cuda_fp16.h"

#define __CPUSIM_T __nv_bfloat16
#define __CPUSIM_T2 __nv_bfloat162
#define __CPUSIM_RAW __nv_bfloat16_raw
#define __CPUSIM_RAW2 __nv_bfloat162_raw
#define __CPUSIM_N bfloat16
#define __CPUSIM_N2 bfloat162
#define __CPUSIM_TOF(bits) __cpusim_fp::bf16_tof(bits)
#define __CPUSIM_FROM(f, mode) __cpusim_fp::bf16_from((f), (mode))
#define __CPUSIM_INF_BITS 0x7f80u
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

typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7f80u)
#define CUDART_NAN_BF16 __ushort_as_bfloat16((unsigned short)0x7fffu)
#define CUDART_MIN_DENORM_BF16 __ushort_as_bfloat16((unsigned short)0x0001u)
#define CUDART_MAX_NORMAL_BF16 __ushort_as_bfloat16((unsigned short)0x7f7fu)
#define CUDART_NEG_ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x8000u)
#define CUDART_ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000u)
#define CUDART_ONE_BF16 __ushort_as_bfloat16((unsigned short)0x3f80u)

#endif
