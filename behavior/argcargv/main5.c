#include <stdio.h>
#include <stdlib.h>

void main(int argc, char* argv[])
{
	printf("void main(int argc, char* argv[])\n");
	printf("argc = %d\n", argc);
	int i = 0;
	for ( ; argv[i]; i++) printf("argv[%d] = \"%s\"\n", i, argv[i]);

	exit(0);
}
