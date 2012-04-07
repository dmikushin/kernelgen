#include <stdio.h>

int counter = 0;

int main(void)
{
	int i;
	for (i = 0; i < 10; i++)
	{
		printf("in loop: counter = %d\n", counter);
		counter++;
	}

	printf("outside loop: counter = %d\n", counter);

	return counter;
}
