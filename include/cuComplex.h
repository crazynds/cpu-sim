/* cuComplex.h: complex numbers as float2 / double2, CUDA's API and its
 * algorithms (division and modulus scaled so that the intermediate products
 * cannot overflow). Usable on host and device. */
#ifndef CPUSIM_CUCOMPLEX_H
#define CPUSIM_CUCOMPLEX_H

#include <math.h>

#include "vector_types.h"

#ifndef __host__
#define __host__
#define __device__
#endif

#define __CPUSIM_CC static inline __host__ __device__

typedef float2 cuFloatComplex;
typedef double2 cuDoubleComplex;
typedef cuFloatComplex cuComplex;

__CPUSIM_CC float cuCrealf(cuFloatComplex x) { return x.x; }
__CPUSIM_CC float cuCimagf(cuFloatComplex x) { return x.y; }
__CPUSIM_CC cuFloatComplex make_cuFloatComplex(float r, float i) {
    cuFloatComplex c;
    c.x = r, c.y = i;
    return c;
}
__CPUSIM_CC cuFloatComplex cuConjf(cuFloatComplex x) { return make_cuFloatComplex(x.x, -x.y); }
__CPUSIM_CC cuFloatComplex cuCaddf(cuFloatComplex x, cuFloatComplex y) { return make_cuFloatComplex(x.x + y.x, x.y + y.y); }
__CPUSIM_CC cuFloatComplex cuCsubf(cuFloatComplex x, cuFloatComplex y) { return make_cuFloatComplex(x.x - y.x, x.y - y.y); }
__CPUSIM_CC cuFloatComplex cuCmulf(cuFloatComplex x, cuFloatComplex y) {
    return make_cuFloatComplex(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
__CPUSIM_CC cuFloatComplex cuCdivf(cuFloatComplex x, cuFloatComplex y) {
    float s = fabsf(y.x) + fabsf(y.y);
    float oos = 1.0f / s;
    float ars = x.x * oos, ais = x.y * oos, brs = y.x * oos, bis = y.y * oos;
    s = brs * brs + bis * bis;
    oos = 1.0f / s;
    return make_cuFloatComplex((ars * brs + ais * bis) * oos, (ais * brs - ars * bis) * oos);
}
__CPUSIM_CC float cuCabsf(cuFloatComplex x) {
    float a = fabsf(x.x), b = fabsf(x.y);
    float v = a > b ? a : b, w = a > b ? b : a;
    float t = w / v;
    t = 1.0f + t * t;
    t = v * sqrtf(t);
    if (v == 0.0f || v > 3.402823466e38f || w > 3.402823466e38f) t = v + w;
    return t;
}

__CPUSIM_CC double cuCreal(cuDoubleComplex x) { return x.x; }
__CPUSIM_CC double cuCimag(cuDoubleComplex x) { return x.y; }
__CPUSIM_CC cuDoubleComplex make_cuDoubleComplex(double r, double i) {
    cuDoubleComplex c;
    c.x = r, c.y = i;
    return c;
}
__CPUSIM_CC cuDoubleComplex cuConj(cuDoubleComplex x) { return make_cuDoubleComplex(x.x, -x.y); }
__CPUSIM_CC cuDoubleComplex cuCadd(cuDoubleComplex x, cuDoubleComplex y) { return make_cuDoubleComplex(x.x + y.x, x.y + y.y); }
__CPUSIM_CC cuDoubleComplex cuCsub(cuDoubleComplex x, cuDoubleComplex y) { return make_cuDoubleComplex(x.x - y.x, x.y - y.y); }
__CPUSIM_CC cuDoubleComplex cuCmul(cuDoubleComplex x, cuDoubleComplex y) {
    return make_cuDoubleComplex(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
__CPUSIM_CC cuDoubleComplex cuCdiv(cuDoubleComplex x, cuDoubleComplex y) {
    double s = fabs(y.x) + fabs(y.y);
    double oos = 1.0 / s;
    double ars = x.x * oos, ais = x.y * oos, brs = y.x * oos, bis = y.y * oos;
    s = brs * brs + bis * bis;
    oos = 1.0 / s;
    return make_cuDoubleComplex((ars * brs + ais * bis) * oos, (ais * brs - ars * bis) * oos);
}
__CPUSIM_CC double cuCabs(cuDoubleComplex x) {
    double a = fabs(x.x), b = fabs(x.y);
    double v = a > b ? a : b, w = a > b ? b : a;
    double t = w / v;
    t = 1.0 + t * t;
    t = v * sqrt(t);
    if (v == 0.0 || v > 1.79769313486231570e308 || w > 1.79769313486231570e308) t = v + w;
    return t;
}

__CPUSIM_CC cuComplex make_cuComplex(float x, float y) { return make_cuFloatComplex(x, y); }
__CPUSIM_CC cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex c) {
    return make_cuDoubleComplex((double)c.x, (double)c.y);
}
__CPUSIM_CC cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex c) { return make_cuFloatComplex((float)c.x, (float)c.y); }
/* x * y + d, in CUDA's order of operations. */
__CPUSIM_CC cuComplex cuCfmaf(cuComplex x, cuComplex y, cuComplex d) {
    float re = x.x * y.x + d.x, im = x.x * y.y + d.y;
    re = -(x.y * y.y) + re;
    im = x.y * y.x + im;
    return make_cuComplex(re, im);
}
__CPUSIM_CC cuDoubleComplex cuCfma(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d) {
    double re = x.x * y.x + d.x, im = x.x * y.y + d.y;
    re = -(x.y * y.y) + re;
    im = x.y * y.x + im;
    return make_cuDoubleComplex(re, im);
}

#undef __CPUSIM_CC
#endif
