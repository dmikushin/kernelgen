#include <stdio.h>

void main(int argc, char* argv[], char* envp[])
{
	printf("void main(int argc, char* argv[])\n");
	printf("argc = %d\n", argc);
	int i = 0;
	for ( ; argv[i]; i++) printf("argv[%d] = \"%s\"\n", i, argv[i]);
	i = 0;
	for ( ; envp[i]; i++) printf("envp[%d] = \"%s\"\n", i, envp[i]);

	return;
}
