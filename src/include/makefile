CI_INCLUDE = $(shell dirname $(shell which nvcc))/../nvvm/ci_include.h

OPT ?= 0

CLANG = clang -cc1 -emit-llvm -fcuda-is-device -triple nvptx64-unknown-unknown

all: kernelgen_runtime.bc kernelgen_monitor.bc kernelgen_cuda.bc

kernelgen_runtime.bc: kernelgen_runtime.cu kernelgen_memory.h kernelgen_interop.h
	$(CLANG) -I. $< -o $@.nopt
	opt -O3 $@.nopt -o $@

kernelgen_monitor.bc: kernelgen_monitor.cu kernelgen_interop.h kernelgen_memory.h kernelgen_runtime.h
	$(CLANG) -I. $< -o $@.nopt
	opt -O3 $@.nopt -o $@

kernelgen_cuda.cpp3.i: kernelgen_cuda.cu
	nvcc -arch=sm_12 -c -keep -v $<

kernelgen_cuda.bc: kernelgen_cuda.cpp3.i
	$(CLANG) -D__NV_POINTER_SIZE=64 -include kernelgen_cuda.h -include $(CI_INCLUDE) $< -o $@.nopt
	opt -O3 $@.nopt -o $@

clean:
	rm -rf *.bc *.bc.nopt *.i *.ii *.gpu *.hash *.fatbin *.cpp *.c *.cubin *.ptx

