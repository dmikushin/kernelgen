#include <stdio.h>

extern "C" const char hello[];

int main()
{
	printf("%s\n", hello);
	return 0;
}

