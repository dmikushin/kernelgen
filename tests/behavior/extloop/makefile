PREFIX = /opt/kgen
DRAGONEGG_FCOMP = dragonegg-gfortran -fplugin=$(PREFIX)/lib64/dragonegg.so
OPT = $(PREFIX)/bin/opt
LLC = $(PREFIX)/bin/llc

.PHONY: sincos.kernels

all: sincos.loop.c sincos.kernels

sincos.kernels: sincos.llvm.cg sincos.loop.cg
	diff -u sincos.llvm.cg sincos.loop.cg

sincos.loop.cg: sincos.loop.s
	$(OPT) -print-function $< -o /dev/null 2>$@

sincos.llvm.cg: sincos.llvm.s
	$(OPT) -print-function $< -o /dev/null 2>$@

sincos.loop.c: sincos.loop.s
	$(OPT) -strip-debug $< | $(LLC) -march=c -o $@

sincos.loop.s: sincos.llvm.s
	$(OPT) -O3 -loop-extract -O1 $< -o $@

sincos.llvm.s: sincos.f90
	$(DRAGONEGG_FCOMP) -g -fplugin-arg-dragonegg-emit-ir -S $< -o $@

clean:
	rm -rf *.s *.c *.cg

