#include <stdlib.h>
#include <stdio.h>
typedef int ElemType;
int main(int argc, char ** argv)
{
   int a = atoi(argv[1]);
   int b = atoi(argv[2]);
   int c = atoi(argv[3]);
   //char * matrix = (char*)malloc(a*b*sizeof(char));
   ElemType * matrix = (ElemType*)malloc((a)*(b + a + c)*sizeof(ElemType));
   //int matrix[a*b*sizeof(int)];
   int i,j;
   for(i = 0; i < a; i++)
       for(j = 0; j < b + c + i; j++)
           matrix[i*(b + c + a) + j] = i+j;
   FILE * file = fopen("result.txt","w");
   fprintf(file, "%d %d %d", matrix[0], matrix[1],matrix[2]);
   return 0;
}
//89099883148x
