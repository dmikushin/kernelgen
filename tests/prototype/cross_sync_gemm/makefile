NVCC = nvcc -g -O3 -arch=sm_20

all: cross_sync_gemm

cross_sync_gemm: cross_sync_gemm.o timing.o
	$(NVCC) cross_sync_gemm.o timing.o -o $@ -lcublas -lrt

cross_sync_gemm.o: cross_sync_gemm.cu
	$(NVCC) -c $< -o $@

timing.o: timing.c
	gcc -c $< -o $@

clean:
	rm -rf cross_sync_gemm *.o

