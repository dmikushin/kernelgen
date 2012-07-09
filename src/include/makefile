CI_INCLUDE = $(shell dirname $(shell which nvcc))/../nvvm/ci_include.h

OPT ?= 0

CLANG = clang -cc1 -emit-llvm -triple ptx64-unknown-unknown -fcuda-is-device

all: kernelgen_runtime.bc kernelgen_monitor.bc kernelgen_cuda.bc

kernelgen_runtime.bc: kernelgen_runtime.cu kernelgen_memory.h kernelgen_interop.h
	$(CLANG) -I. $< -o $@.nopt
	opt -O3 $@.nopt -o $@

kernelgen_monitor.bc: kernelgen_monitor.cu kernelgen_interop.h kernelgen_memory.h kernelgen_runtime.h
	$(CLANG) -I. $< -o $@.nopt
	opt -O3 $@.nopt -o $@

kernelgen_cuda.cpp3.i: kernelgen_cuda.cu
	nvcc -c -keep -v $<

kernelgen_cuda.bc: kernelgen_cuda.cpp3.i
	clang -cc1 -emit-llvm -D__NV_POINTER_SIZE=64 -include $(CI_INCLUDE) -emit-llvm $< -o $@.nopt
	opt -O3 $@.nopt -o $@

clean:
	rm -rf *.bc *.bc.nopt *.i *.ii *.gpu *.hash *.fatbin *.cpp *.c *.cubin *.ptx

