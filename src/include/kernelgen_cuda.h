// XXX: These intrinsics are named starting with __builtin,
//      which is not allowed by Clang, and does not get better,
//      if overcomed. So, rename them.

#define __builtin_log2f  __nvvm_builtin_log2f
#define __builtin_sinf   __nvvm_builtin_sinf
#define __builtin_cosf   __nvvm_builtin_cosf
#define __builtin_round  __nvvm_builtin_round
#define __builtin_roundf __nvvm_builtin_roundf

// XXX: These intrinsics are NYI by LLVM NVPTX backend,
//      so here we just define stubs to replace them, in order
//      to chill out clang errors.

#if (__NV_POINTER_SIZE == 32)
static __attribute__((device)) __attribute__((always_inline))
void __nvvm_memcpy(const unsigned char* a, const unsigned char* b, unsigned int c, int d)
{
}
static __attribute__((device)) __attribute__((always_inline))
void __nvvm_memset(const unsigned char* a, const unsigned char b, unsigned int c, int d)
{
}
#endif

#if (__NV_POINTER_SIZE == 64)
static __attribute__((device)) __attribute__((always_inline))
void __nvvm_memcpy(const unsigned char* a, const unsigned char* b, unsigned long long c, int d)
{
}
static __attribute__((device)) __attribute__((always_inline))
void __nvvm_memset(const unsigned char* a, const unsigned char b, unsigned long long c, int d)
{
}
#endif

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_add_gen_i(volatile int* a, int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
long long __nvvm_atom_add_gen_ll(volatile long long* a, long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
float __nvvm_atom_add_gen_f(volatile float* a, float b)
{
	return 0.0f;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_xchg_gen_i(volatile int* a, int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
long long __nvvm_atom_xchg_gen_ll(volatile long long* a, long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_min_gen_i(volatile int* a, int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_min_gen_ll(volatile long long* a, long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
unsigned int __nvvm_atom_min_gen_ui(volatile unsigned int* a, unsigned int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_min_gen_ull(volatile unsigned long long* a, unsigned long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_max_gen_i(volatile int* a, int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_max_gen_ll(volatile long long* a, long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
unsigned int __nvvm_atom_max_gen_ui(unsigned int* a, unsigned int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_max_gen_ull(volatile unsigned long long* a, unsigned long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
unsigned int __nvvm_atom_inc_gen_ui(unsigned int* a, unsigned int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
unsigned int __nvvm_atom_dec_gen_ui(unsigned int* a, unsigned int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_cas_gen_i(volatile int *p, int compare, int val)
{
        int ret;
        asm volatile (
                "atom.cas.b32    %0, [%1], %2, %3; \n\t"
                : "=r"(ret) : "l"(p), "r"(compare), "r"(val)
        );
        return ret;
}

static __attribute__((device)) __attribute__((always_inline))
long long __nvvm_atom_cas_gen_ll(volatile long long* a, long long b, long long c)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_and_gen_i(volatile int* a, int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_and_gen_ll(volatile long long int* a, long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_or_gen_i(volatile int* a, int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
unsigned int __nvvm_atom_or_gen_ll(volatile long long* a, long long b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_xor_gen_i(volatile int* a, int b)
{
	return 0;
}

static __attribute__((device)) __attribute__((always_inline))
int __nvvm_atom_xor_gen_ll(volatile long long* a, long long b)
{
	return 0;
}

