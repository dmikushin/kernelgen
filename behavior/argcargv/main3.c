#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[], char* envp[])
{
	printf("int main(int argc, char* argv[])\n");
	printf("argc = %d\n", argc);
	int i = 0;
	for ( ; argv[i]; i++) printf("argv[%d] = \"%s\"\n", i, argv[i]);
	i = 0;
	for ( ; envp[i]; i++) if (!strncmp(envp[i], "HOME=", 5)) printf("%s\n", envp[i]);

	return 0;
}
