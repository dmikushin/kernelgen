void matmul(int n, float* a, float* b, float* c)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			c[i * n + j] = 0.0f;
			for (int k = 0; k < n; k++)
				c[i * n + j] = c[i * n + j] + a[i * n + k] * b[k * n + j];
		}
	}
}

