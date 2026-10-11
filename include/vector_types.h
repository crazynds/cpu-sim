/* CUDA built-in vector types (char1..double4, dim3). */
#ifndef CPUSIM_VECTOR_TYPES_H
#define CPUSIM_VECTOR_TYPES_H

#if defined(__CUDA__)
#define __CPUSIM_HD__ __attribute__((host)) __attribute__((device))
#else
#define __CPUSIM_HD__
#endif

#define __CPUSIM_VEC1(name, T, A) struct __attribute__((aligned(A))) name { T x; };
#define __CPUSIM_VEC2(name, T, A) struct __attribute__((aligned(A))) name { T x, y; };
#define __CPUSIM_VEC3(name, T) struct name { T x, y, z; };
#define __CPUSIM_VEC4(name, T, A) struct __attribute__((aligned(A))) name { T x, y, z, w; };

#define __CPUSIM_VECS(base, T, s)          \
    __CPUSIM_VEC1(base##1, T, s)           \
    __CPUSIM_VEC2(base##2, T, 2 * s)       \
    __CPUSIM_VEC3(base##3, T)              \
    __CPUSIM_VEC4(base##4, T, 4 * s)

__CPUSIM_VECS(char, signed char, 1)
__CPUSIM_VECS(uchar, unsigned char, 1)
__CPUSIM_VECS(short, short, 2)
__CPUSIM_VECS(ushort, unsigned short, 2)
__CPUSIM_VECS(int, int, 4)
__CPUSIM_VECS(uint, unsigned int, 4)
__CPUSIM_VECS(float, float, 4)
__CPUSIM_VECS(longlong, long long, 8)
__CPUSIM_VECS(ulonglong, unsigned long long, 8)
/* double4/long4 are 16-byte aligned in CUDA 13 (the 32-byte variants have _32a names). */
__CPUSIM_VEC1(double1, double, 8)
__CPUSIM_VEC2(double2, double, 16)
__CPUSIM_VEC3(double3, double)
__CPUSIM_VEC4(double4, double, 16)
__CPUSIM_VEC1(long1, long, 8)
__CPUSIM_VEC2(long2, long, 16)
__CPUSIM_VEC3(long3, long)
__CPUSIM_VEC4(long4, long, 16)
__CPUSIM_VEC1(ulong1, unsigned long, 8)
__CPUSIM_VEC2(ulong2, unsigned long, 16)
__CPUSIM_VEC3(ulong3, unsigned long)
__CPUSIM_VEC4(ulong4, unsigned long, 16)

#undef __CPUSIM_VECS
#undef __CPUSIM_VEC1
#undef __CPUSIM_VEC2
#undef __CPUSIM_VEC3
#undef __CPUSIM_VEC4

struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    __CPUSIM_HD__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __CPUSIM_HD__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __CPUSIM_HD__ constexpr operator uint3() const { return uint3{x, y, z}; }
#endif
};

#ifndef __cplusplus
typedef struct dim3 dim3;
#endif

#ifdef __cplusplus
#define __CPUSIM_MAKE2(T, B) static inline __CPUSIM_HD__ T##2 make_##T##2(B x, B y) { return T##2{x, y}; }
#define __CPUSIM_MAKE3(T, B) static inline __CPUSIM_HD__ T##3 make_##T##3(B x, B y, B z) { return T##3{x, y, z}; }
#define __CPUSIM_MAKE4(T, B) \
    static inline __CPUSIM_HD__ T##4 make_##T##4(B x, B y, B z, B w) { return T##4{x, y, z, w}; }
#define __CPUSIM_MAKE(T, B) __CPUSIM_MAKE2(T, B) __CPUSIM_MAKE3(T, B) __CPUSIM_MAKE4(T, B)
__CPUSIM_MAKE(char, signed char)
__CPUSIM_MAKE(uchar, unsigned char)
__CPUSIM_MAKE(short, short)
__CPUSIM_MAKE(ushort, unsigned short)
__CPUSIM_MAKE(int, int)
__CPUSIM_MAKE(uint, unsigned int)
__CPUSIM_MAKE(float, float)
__CPUSIM_MAKE(double, double)
__CPUSIM_MAKE(longlong, long long)
__CPUSIM_MAKE(ulonglong, unsigned long long)
#undef __CPUSIM_MAKE
#undef __CPUSIM_MAKE2
#undef __CPUSIM_MAKE3
#undef __CPUSIM_MAKE4
#endif

#endif
