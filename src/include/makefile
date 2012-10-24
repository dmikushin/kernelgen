OPT ?= 0

CLANG = clang -cc1 -emit-llvm -fcuda-is-device -triple nvptx64-unknown-unknown

all: kernelgen_runtime.bc kernelgen_monitor.bc

kernelgen_runtime.bc: kernelgen_runtime.cu kernelgen_memory.h kernelgen_interop.h
	$(CLANG) -I. $< -o $@.nopt
	opt -O3 $@.nopt -o $@

kernelgen_monitor.bc: kernelgen_monitor.cu kernelgen_interop.h kernelgen_memory.h kernelgen_runtime.h
	$(CLANG) -I. $< -o $@.nopt
	opt -O3 $@.nopt -o $@

clean:
	rm -rf *.bc.nopt kernelgen_runtime.bc kernelgen_monitor.bc

