NVCC = nvcc -g -O0 -arch=sm_20 -DVERBOSE

all: cross_sync

cross_sync: cross_sync.cu
	$(NVCC) $< -o $@

clean:
	rm -rf cross_sync

