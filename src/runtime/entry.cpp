#include "runtime.h"

#include <stdlib.h>

// Kernelgen main entry.
extern char* __kernelgen_main;

// Regular main entry.
extern "C" int __regular_main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	char* runmode = getenv("kernelgen_runmode");
	if (runmode)
	{
		int szargs[2] = { sizeof(int), sizeof(char**) };
		return kernelgen_launch(__kernelgen_main, 2, szargs, argc, argv);
	}
	return __regular_main(argc, argv);
}

