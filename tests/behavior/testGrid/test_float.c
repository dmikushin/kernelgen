#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
typedef float ElemType;
int main(int argc, char **argv)
{
	FILE *normal_output = fopen(argv[1],"r");
	FILE *kernelgen_output = fopen(argv[2],"r");
	
	if(!normal_output || !kernelgen_output) {
		printf("!!!!!!!!!!!!!!!------- No data --------!!!!!!!!!!!!!!!\n");
		return -1;
	}
	
	struct stat normal_stat, kernelgen_stat;
	if (stat(argv[1], &normal_stat) == -1 || stat(argv[2], &kernelgen_stat)) {
		printf("!!!!!!!!!!!!!!!------- Can not get file stats --------!!!!!!!!!!!!!!!\n");
        return -1;
    }
	long length1 = normal_stat.st_size;
	long length2 = kernelgen_stat.st_size;
	
	int i = 0;
        int readed;
        ElemType max;
	if(length1 == length2) {
		char * buffer1 = (char *)malloc(length1);
		char * buffer2 = (char *)malloc(length1);
		if(!buffer1 || !buffer2) {
			printf("%s","\n!!!!!!! ---- Can not allocate memory --- !!!!!!!\n");
			return -1;
		}
		if( (readed = fread(buffer1,1,length1,normal_output)) ==
		   fread(buffer2,1,length1,kernelgen_output) && (readed == length1)) {
                   {
                        ElemType *fBuffer1 = (ElemType *)buffer1;
                        ElemType *fBuffer2 = (ElemType *)buffer2;
                        max = abs(fBuffer1[0] - fBuffer2[0]);
			for(i = 0 ; i < length1 / sizeof(ElemType); i++) {
				if( fabs(fBuffer1[i] - fBuffer2[i]) > max) max = fabs(fBuffer1[i] - fBuffer2[i]);
                              //printf("%f %f %f\n",fBuffer1[i],fBuffer2[i],  fabs(fBuffer1[i] - fBuffer2[i]) );
                             }
                   }
		} else {
			printf("!!!!!!!!!!!!!!!------- Can not read files --------!!!!!!!!!!!!!!!\n");
			return 0;
		}
	}
	if(i == length1 / sizeof(ElemType))
		printf("!!!!!!!!!!---- %.20f -----!!!!!!!!!!!\n", max);
	else
		printf("!!!!!!!!!!!!!!!!--------- NO ---------!!!!!!!!!!!!!!!!\n");

	return 0;
}

/*while(!feof(normal_output) && !feof(kernelgen_output))
{
 c1 = fgetc(normal_output);
 c2 = fgetc(kernelgen_output);
 if(c1 != c2) break;
// else putchar(c1);
}*/
//if(feof(normal_output) && feof(kernelgen_output))
