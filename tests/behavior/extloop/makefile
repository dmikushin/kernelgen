PREFIX = /opt/kgen
DRAGONEGG_FCOMP = dragonegg-gfortran -fplugin=$(PREFIX)/lib64/dragonegg.so
OPT = $(PREFIX)/bin/opt
LLC = $(PREFIX)/bin/llc

all: sincos.loop.c

sincos.loop.c: sincos.loop.s
	$(LLC) -march=c $< -o $@

sincos.loop.s: sincos.llvm.s
	$(OPT) -loop-extract $< -o $@

sincos.llvm.s: sincos.f90
	$(DRAGONEGG_FCOMP) -fplugin-arg-dragonegg-emit-ir -S $< -o $@

clean:
	rm -rf *.s *.c

