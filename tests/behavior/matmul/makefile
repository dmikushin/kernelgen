PREFIX = /opt/kernelgen
FCOMP = gfortran
DRAGONEGG_FCOMP = kernelgen-gfortran -fplugin=$(PREFIX)/lib/dragonegg.so
OPT = $(PREFIX)/bin/opt -load $(PREFIX)/lib/LLVMPolly.so
LLC = $(PREFIX)/bin/llc

all: matmul.regular matmul.normalopt matmul.polly

matmul.regular: matmul.F90
	$(FCOMP) $(PRINT) -O3 $< -o $@

matmul.F90.llvm.s: matmul.F90
	$(DRAGONEGG_FCOMP) $(PRINT) -fplugin-arg-dragonegg-emit-ir -S $< -o $@

matmul.F90.preopt.ll: matmul.F90.llvm.s
	$(OPT) -basicaa -mem2reg -simplifycfg -instcombine -tailcallelim -loop-simplify \
        -lcssa -loop-rotate -lcssa -loop-unswitch -instcombine -loop-simplify -lcssa -indvars -loop-deletion \
	-instcombine -polly-prepare -polly-region-simplify -indvars -S $< -o $@

matmul.polly.ll: matmul.F90.preopt.ll
	$(OPT) -basicaa -polly-optimize-isl -polly-codegen $< | $(OPT) -O3 > $@

matmul.normalopt.ll: matmul.F90.llvm.s
	$(OPT) -O3 $< -o $@

matmul.polly.s: matmul.polly.ll
	$(LLC) $< -o $@

matmul.normalopt.s: matmul.normalopt.ll
	$(LLC) $< -o $@

matmul.polly: matmul.polly.s
	$(FCOMP) $< -o $@

matmul.normalopt: matmul.normalopt.s
	$(FCOMP) $< -o $@

benchmark: matmul.regular matmul.normalopt matmul.polly
	./matmul.polly
	./matmul.normalopt
	./matmul.regular

clean:
	rm -rf  matmul.regular matmul.normalopt matmul.polly* *.ll *.s

