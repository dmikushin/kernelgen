#include <kernelgen.h>

extern kernelgen_kernel_config_t sincos_loop_1_kernelgen_config;

extern "C" void sincos_loop_1_kernelgen_init_deps(kernelgen_kernel_config_t* config);

__attribute__ ((__constructor__(102))) void sincos_loop_1_kernelgen_init()
{
kernelgen_kernel_init(&sincos_loop_1_kernelgen_config, 1, 1, "sincos", 6, 0);
sincos_loop_1_kernelgen_init_deps(&sincos_loop_1_kernelgen_config);
}

__attribute__ ((__destructor__(102))) void sincos_loop_1_kernelgen_free()
{
kernelgen_kernel_free_deps(&sincos_loop_1_kernelgen_config);
kernelgen_kernel_free(&sincos_loop_1_kernelgen_config);
}
