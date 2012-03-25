#ifdef __OPENCL_DEVICE_FUNC__
#ifdef __OPENCL_HAVE_DOUBLE__
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#define __GNUG__
#define size_t size_t
#endif

/* Provide Declarations */
#ifndef __OPENCL_DEVICE_FUNC__
#include <stdarg.h>
#endif // __OPENCL_DEVICE_FUNC__
#include <setjmp.h>
#include <limits.h>
/* get a declaration for alloca */
#if defined(__CYGWIN__) || defined(__MINGW32__)
#define  alloca(x) __builtin_alloca((x))
#define _alloca(x) __builtin_alloca((x))
#elif defined(__APPLE__)
extern void *__builtin_alloca(unsigned long);
#define alloca(x) __builtin_alloca(x)
#define longjmp _longjmp
#define setjmp _setjmp
#elif defined(__sun__)
#if defined(__sparcv9)
extern void *__builtin_alloca(unsigned long);
#else
extern void *__builtin_alloca(unsigned int);
#endif
#define alloca(x) __builtin_alloca(x)
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__) || defined(__arm__)
#define alloca(x) __builtin_alloca(x)
#elif defined(_MSC_VER)
#define inline _inline
#define alloca(x) _alloca(x)
#elif defined(__CUDA_DEVICE_FUNC__)
#define alloca(x) kernelgen_malloc(x)
#else
#include <alloca.h>
#endif

#ifndef __GNUC__  /* Can only support "linkonce" vars with GCC */
#define __attribute__(X)
#endif

#if defined(__GNUC__) && defined(__APPLE_CC__)
#define __EXTERNAL_WEAK__ __attribute__((weak_import))
#elif defined(__GNUC__)
#define __EXTERNAL_WEAK__ __attribute__((weak))
#else
#define __EXTERNAL_WEAK__
#endif

#if defined(__GNUC__) && defined(__APPLE_CC__)
#define __ATTRIBUTE_WEAK__
#elif defined(__GNUC__)
#define __ATTRIBUTE_WEAK__ __attribute__((weak))
#else
#define __ATTRIBUTE_WEAK__
#endif

#if defined(__GNUC__)
#define __HIDDEN__ __attribute__((visibility("hidden")))
#endif

#ifdef __GNUC__
#define LLVM_NAN(NanStr)   __builtin_nan(NanStr)   /* Double */
#define LLVM_NANF(NanStr)  __builtin_nanf(NanStr)  /* Float */
#define LLVM_NANS(NanStr)  __builtin_nans(NanStr)  /* Double */
#define LLVM_NANSF(NanStr) __builtin_nansf(NanStr) /* Float */
#define LLVM_INF           __builtin_inf()         /* Double */
#define LLVM_INFF          __builtin_inff()        /* Float */
#define LLVM_PREFETCH(addr,rw,locality) __builtin_prefetch(addr,rw,locality)
#define __ATTRIBUTE_CTOR__ __attribute__((constructor))
#define __ATTRIBUTE_DTOR__ __attribute__((destructor))
#define LLVM_ASM           __asm__
#else
#define LLVM_NAN(NanStr)   ((double)0.0)           /* Double */
#define LLVM_NANF(NanStr)  0.0F                    /* Float */
#define LLVM_NANS(NanStr)  ((double)0.0)           /* Double */
#define LLVM_NANSF(NanStr) 0.0F                    /* Float */
#define LLVM_INF           ((double)0.0)           /* Double */
#define LLVM_INFF          0.0F                    /* Float */
#define LLVM_PREFETCH(addr,rw,locality)            /* PREFETCH */
#define __ATTRIBUTE_CTOR__
#define __ATTRIBUTE_DTOR__
#define LLVM_ASM(X)
#endif

/* Old GCC's, or compilers not GCC */
#if defined(__CUDA_DEVICE_FUNC__) || defined(__OPENCL_DEVICE_FUNC__) || (__GNUC__ < 4)
#define __builtin_stack_save() 0   /* not implemented */
#define __builtin_stack_restore(X) /* noop */
#endif

#if __GNUC__ && __LP64__ && !__CUDA_DEVICE_FUNC__ /* 128-bit integer types */
typedef int __attribute__((mode(TI))) llvmInt128;
typedef unsigned __attribute__((mode(TI))) llvmUInt128;
#endif

#define CODE_FOR_MAIN() /* Any target-specific code for main()*/

#if !defined(__cplusplus) && !defined(__OPENCL_DEVICE_FUNC__)
typedef unsigned char bool;
#endif


/* Support for floating point constants */
typedef unsigned long long ConstantDoubleTy;
typedef unsigned int        ConstantFloatTy;
typedef struct { unsigned long long f1; unsigned short f2; unsigned short pad[3]; } ConstantFP80Ty;
typedef struct { unsigned long long f1; unsigned long long f2; } ConstantFP128Ty;


/* Global Declarations */
/* Helper union for bitcasts */
typedef union {
  unsigned int Int32;
  unsigned long long Int64;
  float Float;
#ifdef __OPENCL_DEVICE_FUNC__
#ifdef __OPENCL_HAVE_DOUBLE__
  double Double;
#endif
#else
  double Double;
#endif
} llvmBitCastUnion;
/* Structure forward decls */
struct l_unnamed_0;
struct l_unnamed_1;
struct l_unnamed_2;
struct l_unnamed_3;

/* Typedefs */
typedef struct l_unnamed_0 l_unnamed_0;
typedef struct l_unnamed_1 l_unnamed_1;
typedef struct l_unnamed_2 l_unnamed_2;
typedef struct l_unnamed_3 l_unnamed_3;

/* Structure contents */
struct

#if defined(__CUDA_DEVICE_FUNC__) || defined(__OPENCL_DEVICE_FUNC__)
__attribute__ ((aligned(16)))
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_0 {
  unsigned char array[14];
};

struct

#if defined(__CUDA_DEVICE_FUNC__) || defined(__OPENCL_DEVICE_FUNC__)
__attribute__ ((aligned(16)))
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_1 {
  unsigned char array[7];
};

struct

#if defined(__CUDA_DEVICE_FUNC__) || defined(__OPENCL_DEVICE_FUNC__)
__attribute__ ((aligned(16)))
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_2 {
  unsigned char array[26];
};

struct

#if defined(__CUDA_DEVICE_FUNC__) || defined(__OPENCL_DEVICE_FUNC__)
__attribute__ ((aligned(16)))
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_3 {
  float array[1];
};


/* External Global Variable Declarations */
#ifdef __CUDA_DEVICE_FUNC__
__attribute__((device))
#endif
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
extern
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_0 _OC_cst;
#ifdef __CUDA_DEVICE_FUNC__
__attribute__((device))
#endif
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
extern
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_1 _OC_cst1;
#ifdef __CUDA_DEVICE_FUNC__
__attribute__((device))
#endif
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
extern
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_2 _OC_cst2;

/* Function Declarations */
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
float sinf(float );
float cosf(float );
void __kernelgen_sincos__loop_12(
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_args);
unsigned long long kernelgen_threadIdx_x(void);
unsigned long long kernelgen_blockIdx_x(void);
unsigned long long kernelgen_threadIdx_y(void);
unsigned long long kernelgen_blockIdx_y(void);
unsigned long long kernelgen_threadIdx_z(void);
unsigned long long kernelgen_blockIdx_z(void);
void abort(void);
#endif // __CUDA_DEVICE_FUNC__


/* Global Variable Declarations */
#ifndef __CUDA_DEVICE_FUNC__
extern l_unnamed_0 _OC_cst __HIDDEN__;
#endif // __CUDA_DEVICE_FUNC__
#ifndef __CUDA_DEVICE_FUNC__
extern l_unnamed_1 _OC_cst1 __HIDDEN__;
#endif // __CUDA_DEVICE_FUNC__
#ifndef __CUDA_DEVICE_FUNC__
extern l_unnamed_2 _OC_cst2 __HIDDEN__;
#endif // __CUDA_DEVICE_FUNC__


/* Global Variable Definitions and Initialization */
#ifdef __CUDA_DEVICE_FUNC__
__attribute__((device))
#endif // __CUDA_DEVICE_FUNC__
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
extern
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_0 _OC_cst __HIDDEN__ = { "../sincos.f90" };
#ifdef __CUDA_DEVICE_FUNC__
__attribute__((device))
#endif // __CUDA_DEVICE_FUNC__
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
extern
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_1 _OC_cst1 __HIDDEN__ = { { ((unsigned char )100), ((unsigned char )105), ((unsigned char )102), ((unsigned char )102), ((unsigned char )32), ((unsigned char )61), ((unsigned char )32) } };
#ifdef __CUDA_DEVICE_FUNC__
__attribute__((device))
#endif // __CUDA_DEVICE_FUNC__
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
extern
#endif // __CUDA_DEVICE_FUNC__
l_unnamed_2 _OC_cst2 __HIDDEN__ = { "Usage: %s <nx> <ny> <nz>\n" };


/* Function Bodies */
#define llvm_fcmp_ord(X, Y) ((X) == (X) && (Y) == (Y))
#define llvm_fcmp_uno(X, Y) ((X) != (X) || (Y) != (Y))
#define llvm_fcmp_ueq(X, Y) ((X) == (Y) || llvm_fcmp_uno(X, Y))
#define llvm_fcmp_une(X, Y) ((X) != (Y))
#define llvm_fcmp_ult(X, Y) ((X) <  (Y) || llvm_fcmp_uno(X, Y))
#define llvm_fcmp_ugt(X, Y) ((X) >  (Y) || llvm_fcmp_uno(X, Y))
#define llvm_fcmp_ule(X, Y) ((X) <= (Y) || llvm_fcmp_uno(X, Y))
#define llvm_fcmp_uge(X, Y) ((X) >= (Y) || llvm_fcmp_uno(X, Y))
#define llvm_fcmp_oeq(X, Y) ((X) == (Y))
#define llvm_fcmp_one(X, Y) ((X) != (Y) && llvm_fcmp_ord(X, Y))
#define llvm_fcmp_olt(X, Y) ((X) <  (Y))
#define llvm_fcmp_ogt(X, Y) ((X) >  (Y))
#define llvm_fcmp_ole(X, Y) ((X) <= (Y))
#define llvm_fcmp_oge(X, Y) ((X) >= (Y))

#ifdef __CUDA_DEVICE_FUNC__
__attribute__((global)) __attribute__((used))
#endif // __CUDA_DEVICE_FUNC__
void __kernelgen_sincos__loop_12(
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_args) {
  unsigned long long llvm_cbe_threadIdx_2e_x;
  unsigned long long llvm_cbe_blockIdx_2e_x;
  unsigned long long llvm_cbe_threadIdx_2e_y;
  unsigned long long llvm_cbe_blockIdx_2e_y;
  unsigned long long llvm_cbe_threadIdx_2e_z;
  unsigned long long llvm_cbe_blockIdx_2e_z;
  unsigned long long llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_z;
  unsigned long long llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_y;
  unsigned long long llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_x;
  unsigned long long llvm_cbe_p_;
  float llvm_cbe__p_scalar_;
  float llvm_cbe_p_28;
  float llvm_cbe__p_scalar_29;
  float llvm_cbe_p_30;

  llvm_cbe_threadIdx_2e_x =  /*tail*/ kernelgen_threadIdx_x();
  llvm_cbe_blockIdx_2e_x =  /*tail*/ kernelgen_blockIdx_x();
  llvm_cbe_threadIdx_2e_y =  /*tail*/ kernelgen_threadIdx_y();
  llvm_cbe_blockIdx_2e_y =  /*tail*/ kernelgen_blockIdx_y();
  llvm_cbe_threadIdx_2e_z =  /*tail*/ kernelgen_threadIdx_z();
  llvm_cbe_blockIdx_2e_z =  /*tail*/ kernelgen_blockIdx_z();
  llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_z = ((unsigned long long )(((unsigned long long )(llvm_cbe_blockIdx_2e_z << 1ull)) + ((unsigned long long )llvm_cbe_threadIdx_2e_z)));
  if ((((signed long long )llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_z) > ((signed long long )63ull))) {
    goto llvm_cbe_CUDA_2e_AfterLoop_2e_z;
  } else {
    goto llvm_cbe_CUDA_2e_LoopHeader_2e_z_2e_preheader;
  }

llvm_cbe_CUDA_2e_LoopHeader_2e_z_2e_preheader:
  llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_y = ((unsigned long long )(((unsigned long long )(llvm_cbe_blockIdx_2e_y << 4ull)) + ((unsigned long long )llvm_cbe_threadIdx_2e_y)));
  llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_x = ((unsigned long long )(((unsigned long long )(llvm_cbe_blockIdx_2e_x << 5ull)) + ((unsigned long long )llvm_cbe_threadIdx_2e_x)));
  if (((((((signed long long )llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_y) > ((signed long long )511ull)) | (((signed long long )llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_x) > ((signed long long )511ull)))&1))) {
    goto llvm_cbe_CUDA_2e_AfterLoop_2e_z;
  } else {
    goto llvm_cbe_CUDA_2e_LoopHeader_2e_y_2e_backedge;
  }

llvm_cbe_CUDA_2e_LoopHeader_2e_y_2e_backedge:
  llvm_cbe_p_ = ((unsigned long long )(((unsigned long long )(((unsigned long long )(((unsigned long long )(llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_y << 9ull)) + ((unsigned long long )(llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_z << 18ull)))))) + ((unsigned long long )llvm_cbe_BlockLB_2e_Add_2e_ThreadPosInBlock_2e_x)));
  llvm_cbe__p_scalar_ = *((&(*((
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_3 (*))(unsigned long)8595607552ull)).array[((signed long long )llvm_cbe_p_)]));
  llvm_cbe_p_28 =  /*tail*/ sinf(llvm_cbe__p_scalar_);
  llvm_cbe__p_scalar_29 = *((&(*((
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_3 (*))(unsigned long)8662720512ull)).array[((signed long long )llvm_cbe_p_)]));
  llvm_cbe_p_30 =  /*tail*/ cosf(llvm_cbe__p_scalar_29);
  *((&(*((
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_3 (*))(unsigned long)8796946432ull)).array[((signed long long )llvm_cbe_p_)])) = (((float )(llvm_cbe_p_28 + llvm_cbe_p_30)));
  goto llvm_cbe_CUDA_2e_AfterLoop_2e_z;

llvm_cbe_CUDA_2e_AfterLoop_2e_z:
  return;
}

