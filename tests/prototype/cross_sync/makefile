all: cross_sync

cross_sync: cross_sync.cu
	nvcc -g -O3 -arch=sm_20 $< -o $@ -lpthread

clean:
	rm -rf cross_sync

