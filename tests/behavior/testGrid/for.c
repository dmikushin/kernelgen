#include <stdlib.h>
#include <stdio.h>
//#include <linux/time.h>
#include <time.h>
#include <math.h>

static double get_diff(struct timespec start,struct timespec stop)
{
	long long seconds = stop.tv_sec - start.tv_sec;
	long long nanoseconds = stop.tv_nsec - start.tv_nsec;

	if (stop.tv_nsec < start.tv_nsec)
	{
		seconds--;
		nanoseconds = (1000000000 - start.tv_nsec) + stop.tv_nsec;
	}

	return (double)0.000000001 * nanoseconds + seconds;
}
#define PI 3.1415926f
typedef float ElemType;
int main(int argc, char ** argv)
{
        struct timespec start;
        struct timespec stop;

        clock_gettime(CLOCK_REALTIME, &start);

        if(argc != 5)
        {
            printf("%s","\n!!!!!!! ---- Not Enough parameters ---- !!!!!!!\n");
            return -1;
        }
        
	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);
        char * file_name = argv[4];	
	
        ElemType * matrix = (ElemType*)malloc(a*b*c*sizeof(ElemType));
	//ElemType matrix[a*b*c];
        
        if(!matrix)
        {
            printf("%s","\n!!!!!!! ---- Can not allocate memory --- !!!!!!!\n");
            return -1;
        } else 
        printf("\n!!!!!!!! ---- %d %d %d %s  ---- !!!!!!!\n", a, b, c, file_name);
        
        int i=0,j=0,k=0,s =0;
        float size = a*b*c;
        for(i = 0 ; i < a; i++)
        for(j = 0; j < b; j++)
        for(k = 0; k < c; k++)
        { 
              int index = i*b*c + j*c + k; 
              float value = sinf (  (float)(index) / ( size / (2.0f*PI)) ) + sinf (  (float)(index) / ( size / (2.0f*PI)));
              matrix[index] = value;
        }

        printf("%s","!!!!!!!! ---- Start Writing of array   ------- !!!!!!!\n");
        FILE * file = fopen(file_name,"w");
        fwrite ( matrix, sizeof(ElemType), a*b*c, file );
        fclose(file);
        printf("%s","!!!!!!!! ---- Program correctly completed ---- !!!!!!!\n");
         
        clock_gettime(CLOCK_REALTIME, &stop);
        printf("!!!!!!!! ---- Total time : %f sec  ---- !!!!!!!!\n\n", get_diff(start,stop));
        return 0;
}
