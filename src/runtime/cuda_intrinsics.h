/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C backend.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 * you must not claim that you wrote the original software.
 * If you use this software in a product, an acknowledgment
 * in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such,
 * and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

// CUDA functions that could be directly resolved on the GPU,
// without host callback.

"abs", "labs", "llabs", "fabs", "fabsf",
"min", "umin", "llmin", "ullmin", "fminf", "fmin",
"max", "umax", "llmax", "ullmax", "fmax", "fmax",
"sin", "sinf",
"cos", "cosf",
"sincos", "sincosf",
"tan", "tanf",
"sqrt", "sqrtf", "rsqrt", "rsqrtf",
"exp", "expf", "exp2", "exp2f", "exp10", "exp10f", "expm1", "expm1f",
"log", "logf", "log2", "log2f", "log10", "log10f", "log1p", "log1pf",
"floor", "floorf",
"cosh", "coshf",
"sinh", "sinhf",
"tanh", "tanhf",
"acosh", "acoshf", "asinh", "asinhf", "atanh", "atanhf",
"ldexp", "ldexpf", "logb", "logbf", "ilogb", "ilogbf",
"scalbn", "scalbnf", "scalbln", "scalblnf",
"frexp", "frexpf",
"round", "roundf", "lround", "lroundf", "llround", "llroundf",
"rint", "rintf", "lrint", "lrintf", "llrint", "llrintf",
"nearbyint", "nearbyintf",
"ceil", "ceilf",
"trunc", "truncf",
"fdim", "fdimf",
"atan", "atanf", "atan2", "atan2f",
"asin", "asinf",
"acos", "acosf",
"hypot", "hypot", "hypotf",
"cbrt", "cbrtf", "rcbrt", "rcbrtf",
"sinpi", "sinpif", "cospi", "cospif",
"pow", "powf",
"modf", "modff",
"fmod", "fmodf",
"remainder", "remainderf", "remquo", "remquof",
"erf", "erff",
"erfinv", "erfinvf", "erfc", "erfcf", "erfcinv", "erfcinvf",
"lgamma", "lgammaf", "tgamma", "tgammaf",
"copysign", "copysignf",
"nextafter", "nextafterf", 
"nan", "nanf",
"isinf", "__isinf", "__isinff", "__isinfl",
"isnan", "__isnan", "__isnanf", "__isnanl",
"isfinite", "__isfinited", "__isfinitef",
"signbit", "__signbit", "__signbitf", "__signbitd", "__signbitl",
"__finite", "__finitef",
"fma", "fmaf",
"abort"

