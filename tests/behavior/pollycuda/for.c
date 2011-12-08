#define N 10

int main(int argc, char ** argv)
{
   int matrix[N][N];
   int i,j;
   for(i = 0; i < N; i++)
       for(j = 0; j < N; j++)
           matrix[i][j] = i+j+200;
   return matrix[0][0];
}
//89099883148
