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

#if __GNUC__ < 4 /* Old GCC's, or compilers not GCC */ 
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
/* Module asm statements */
asm("\t.ident\t\"GCC: (GNU) 4.5.4 20110810 (prerelease) LLVM: 136347M\"\n"
"");
/* End Module asm statements */
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

/* Typedefs */
typedef struct l_unnamed_0 l_unnamed_0;

/* Structure contents */
struct l_unnamed_0 {
  float array[1];
};


/* Function Declarations */
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
#endif // __CUDA_DEVICE_FUNC__
#ifdef __CUDA_DEVICE_FUNC__
extern "C" __global__
#endif
#define sincos_loop_1_kernelgen_ sincos_loop_1_kernelgen_cuda
void sincos_loop_1_kernelgen_(
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_nz, 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_ny, 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_nx, 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_0 (*llvm_cbe_xy), 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_0 (*llvm_cbe_x), 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_0 (*llvm_cbe_y));
#ifdef __CUDA_DEVICE_FUNC__
__device__
#endif
static void sincos_loop_1_kernelgen_blockidx_z(unsigned int* index, unsigned int start, unsigned int end) { *index = threadIdx.z + start; }
#ifdef __CUDA_DEVICE_FUNC__
__device__
#endif
static void sincos_loop_1_kernelgen_blockidx_y(unsigned int* index, unsigned int start, unsigned int end) { *index = blockIdx.y + start; }
#ifdef __CUDA_DEVICE_FUNC__
__device__
#endif
static void sincos_loop_1_kernelgen_blockidx_x(unsigned int* index, unsigned int start, unsigned int end) { *index = blockIdx.x + start; }
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
float sinf(float );
#endif // __CUDA_DEVICE_FUNC__ && __OPENCL_DEVICE_FUNC__
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
float cosf(float );
#endif // __CUDA_DEVICE_FUNC__ && __OPENCL_DEVICE_FUNC__
#if !defined(__CUDA_DEVICE_FUNC__) && !defined(__OPENCL_DEVICE_FUNC__)
void abort(void);
#endif // __CUDA_DEVICE_FUNC__ && __OPENCL_DEVICE_FUNC__


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

void sincos_loop_1_kernelgen_(
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_nz, 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_ny, 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
unsigned int *llvm_cbe_nx, 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_0 (*llvm_cbe_xy), 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_0 (*llvm_cbe_x), 
#ifdef __OPENCL_DEVICE_FUNC__
__global
#endif // __OPENCL_DEVICE_FUNC__
l_unnamed_0 (*llvm_cbe_y)) {
  unsigned int llvm_cbe_memtmp;    /* Address-exposed local */
  unsigned int llvm_cbe_memtmp3;    /* Address-exposed local */
  unsigned int llvm_cbe_memtmp4;    /* Address-exposed local */
  unsigned int llvm_cbe_tmp__1;
  unsigned long long llvm_cbe_tmp__2;
  unsigned long long llvm_cbe_tmp__3;
  unsigned int llvm_cbe_tmp__4;
  unsigned long long llvm_cbe_tmp__5;
  unsigned long long llvm_cbe_tmp__6;
  unsigned int llvm_cbe_tmp__7;
  unsigned int llvm_cbe_tmp__8;
  unsigned int llvm_cbe_tmp__9;
  unsigned int llvm_cbe_tmp__10;
  unsigned int llvm_cbe_tmp__11;
  unsigned int llvm_cbe_tmp__12;
  unsigned long long llvm_cbe_tmp__13;
  float llvm_cbe_tmp__14;
  float llvm_cbe_tmp__15;
  float llvm_cbe_tmp__16;
  float llvm_cbe_tmp__17;

  llvm_cbe_tmp__1 = *llvm_cbe_nx;
  llvm_cbe_tmp__2 = ((signed long long )(signed int )llvm_cbe_tmp__1);
  llvm_cbe_tmp__3 = (((((signed long long )llvm_cbe_tmp__2) < ((signed long long )0ull))) ? (0ull) : (llvm_cbe_tmp__2));
  llvm_cbe_tmp__4 = *llvm_cbe_ny;
  llvm_cbe_tmp__5 = ((unsigned long long )(((unsigned long long )llvm_cbe_tmp__3) * ((unsigned long long )(((signed long long )(signed int )llvm_cbe_tmp__4)))));
  llvm_cbe_tmp__6 = (((((signed long long )llvm_cbe_tmp__5) < ((signed long long )0ull))) ? (0ull) : (llvm_cbe_tmp__5));
  llvm_cbe_tmp__7 = *llvm_cbe_nz;
  sincos_loop_1_kernelgen_blockidx_z((&llvm_cbe_memtmp), 1u, llvm_cbe_tmp__7);
  llvm_cbe_tmp__8 = *llvm_cbe_ny;
  sincos_loop_1_kernelgen_blockidx_y((&llvm_cbe_memtmp3), 1u, llvm_cbe_tmp__8);
  llvm_cbe_tmp__9 = *llvm_cbe_nx;
  sincos_loop_1_kernelgen_blockidx_x((&llvm_cbe_memtmp4), 1u, llvm_cbe_tmp__9);
  llvm_cbe_tmp__10 = *(&llvm_cbe_memtmp4);
  llvm_cbe_tmp__11 = *(&llvm_cbe_memtmp);
  llvm_cbe_tmp__12 = *(&llvm_cbe_memtmp3);
  llvm_cbe_tmp__13 = ((unsigned long long )(((unsigned long long )(((unsigned long long )(((unsigned long long )(((unsigned long long )(((unsigned long long )(((signed long long )(signed int )llvm_cbe_tmp__10))) + ((unsigned long long )(((unsigned long long )(((unsigned long long )(llvm_cbe_tmp__3 ^ 18446744073709551615ull)) - ((unsigned long long )llvm_cbe_tmp__6))))))))) + ((unsigned long long )(((unsigned long long )(((unsigned long long )(((signed long long )(signed int )llvm_cbe_tmp__11))) * ((unsigned long long )llvm_cbe_tmp__6))))))))) + ((unsigned long long )(((unsigned long long )(((unsigned long long )(((signed long long )(signed int )llvm_cbe_tmp__12))) * ((unsigned long long )llvm_cbe_tmp__3)))))));
  llvm_cbe_tmp__14 = *((&(*llvm_cbe_x).array[((signed long long )llvm_cbe_tmp__13)]));
  llvm_cbe_tmp__15 = sinf(llvm_cbe_tmp__14);
  llvm_cbe_tmp__16 = *((&(*llvm_cbe_y).array[((signed long long )llvm_cbe_tmp__13)]));
  llvm_cbe_tmp__17 = cosf(llvm_cbe_tmp__16);
  *((&(*llvm_cbe_xy).array[((signed long long )llvm_cbe_tmp__13)])) = (((float )(llvm_cbe_tmp__15 + llvm_cbe_tmp__17)));
  return;
}

