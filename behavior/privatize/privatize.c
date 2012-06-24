#include <stdio.h>

int counter = 0;
int n = 10;

int main(void)
{
	int i;
	for (i = 0; i < n; i++)
	{
		printf("in loop: counter = %d\n", counter);
		counter++;
	}

	printf("outside loop: counter = %d\n", counter);

	return 0;
}
