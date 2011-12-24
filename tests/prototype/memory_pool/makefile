all: memory_pool

memory_pool: memory_pool.cu kernelgen_memory.h
	nvcc -g -G $< -o $@

clean:
	rm -rf memory_pool

