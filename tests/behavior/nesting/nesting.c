#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		printf("Usage: %s <n>\n", argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	float a[n][n][n][n];
	for (int i1 = 0; i1 < n; i1++)
		for (int i2 = 0; i2 < n; i2++)
			for (int i3 = 0; i3 < n; i3++)
				for (int i4 = 0; i4 < n; i4++)
					a[i1][i2][i3][i4] = i4;

	return 0;
}

