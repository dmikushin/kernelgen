all: cuda_mmap1 cuda_mmap2

cuda_mmap1: cuda_mmap1.cu
	nvcc -g $< -o $@

cuda_mmap2: cuda_mmap2.cu
	nvcc -g $< -o $@ -lcuda

clean:
	rm -rf cuda_mmap1 cuda_mmap2

