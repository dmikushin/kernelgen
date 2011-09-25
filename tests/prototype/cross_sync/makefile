all: cross_sync

cross_sync: cross_sync.cu
	nvcc -g -O0 -arch=sm_20 $< -o $@

clean:
	rm -rf cross_sync

