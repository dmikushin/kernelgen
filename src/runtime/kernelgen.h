#include <string>
#include <map>

#define KERNELGEN_RUNMODE_COUNT 3

extern int kernelgen_runmode;

typedef struct
{
	// References to tables of compiled kernels
	// for each supported runmode.
	std::map<std::string, char*> binary[KERNELGEN_RUNMODE_COUNT];
	
	// Kernel LLVM IR source code.
	char* source;
}
kernelgen_kernel_t;

extern std::map<std::string, kernelgen_kernel_t*> kernelgen_kernels;

// Perform initial kernelgen configuration.
int kernelgen_init();

// Launch kernel from the spepcified source code address.
int kernelgen_launch(char* kernel, int nargs, ...);

