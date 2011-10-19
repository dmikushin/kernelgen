#include "kernelgen.h"

extern char* __kernelgen_main;

extern "C" int __regular_main(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	/*char* runmode = getenv("kernelgen_runmode");
	if (runmode)
		return kernelgen_launch(__kernelgen_main);*/
	return __regular_main(argc, argv);
}

