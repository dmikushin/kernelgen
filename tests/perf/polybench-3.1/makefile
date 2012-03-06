.PHONY: atax

all: atax

FATAX = -I./utilities -I./linear-algebra/kernels/atax

atax: atax_base atax_time atax_ref

atax_base: polybench.o atax_base.o
	kernelgen $^ -o $@

atax_time: polybench.o atax_time.o
	kernelgen $^ -o $@

atax_ref: polybench.o atax_ref.o
	kernelgen $^ -o $@

atax_base.o: linear-algebra/kernels/atax/atax.c
	kernelgen $(FATAX) -c $< -o $@

atax_time.o: linear-algebra/kernels/atax/atax.c
	kernelgen -DPOLYBENCH_TIME $(FATAX) -c $< -o $@

atax_ref.o: linear-algebra/kernels/atax/atax.c
	kernelgen -O0 -DPOLYBENCH_DUMP_ARRAYS $(FATAX) -c $< -o $@

polybench.o: utilities/polybench.c
	kernelgen -c $< -o $@

clean:
	rm -rf *.o atax_base atax_time atax_ref

