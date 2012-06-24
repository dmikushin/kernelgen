#include <stdio.h>

int main(int argc, char* argv[])
{
	printf("int main(int argc, char* argv[])\n");
	printf("argc = %d\n", argc);
	int i = 0;
	for ( ; argv[i]; i++) printf("argv[%d] = \"%s\"\n", i, argv[i]);

	return 0;
}
