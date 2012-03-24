#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
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
	if(length1 == length2) {
		char * buffer1 = (char *)malloc(length1);
		char * buffer2 = (char *)malloc(length1);
		if(!buffer1 || !buffer2) {
			printf("%s","\n!!!!!!! ---- Can not allocate memory --- !!!!!!!\n");
			return -1;
		}
		if(fread(buffer1,length1,1,normal_output) ==
		   fread(buffer2,length1,1,kernelgen_output)) {
			for(i = 0 ; i < length1; i++)
				if(buffer1[i] != buffer2[i]) break;
		} else {
			printf("!!!!!!!!!!!!!!!------- Can not read files --------!!!!!!!!!!!!!!!\n");
			return 0;
		}
	}
	if(i == length1)
		printf("!!!!!!!!!!!!!!!!--------- OK ---------!!!!!!!!!!!!!!!!\n");
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
