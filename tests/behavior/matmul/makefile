PREFIX = /opt/kgen
FCOMP = gfortran
DRAGONEGG_FCOMP = dragonegg-gfortran -fplugin=$(PREFIX)/lib64/dragonegg.so
OPT = $(PREFIX)/bin/opt # -load $(PREFIX)/lib/LLVMPolly.so
LLC = $(PREFIX)/bin/llc
#PRINT = -DPRINT

all: matmul.regular matmul.normalopt matmul.polly

matmul.regular: matmul.F90
	$(FCOMP) $(PRINT) -O3 $< -o $@

matmul.F90.llvm.s: matmul.F90
	$(DRAGONEGG_FCOMP) $(PRINT) -fplugin-arg-dragonegg-emit-ir -S $< -o $@

matmul.F90.preopt.ll: matmul.F90.llvm.s
	$(OPT) -basicaa -mem2reg -simplify-libcalls -simplifycfg -instcombine -tailcallelim -loop-simplify \
	-lcssa -loop-rotate -lcssa -loop-unswitch -instcombine -loop-simplify -lcssa -indvars -loop-deletion \
	-instcombine -polly-prepare -polly-region-simplify -indvars -S $< -o $@

matmul.polly.ll: matmul.F90.preopt.ll
	$(OPT) -basicaa -polly-optimize-isl -polly-codegen $< | $(OPT) -O3 > $@

matmul.normalopt.ll: matmul.F90.preopt.ll
	$(OPT) $< | $(OPT) -O3 > $@

matmul.polly.s: matmul.polly.ll
	$(LLC) $< -o $@

matmul.normalopt.s: matmul.normalopt.ll
	$(LLC) $< -o $@

matmul.polly: matmul.polly.s
	$(FCOMP) $< -o $@

matmul.normalopt: matmul.normalopt.s
	$(FCOMP) $< -o $@

benchmark: matmul.regular matmul.normalopt matmul.polly
	@echo "time ./matmul.polly"
	time -f "%E real, %U user, %S sys" ./matmul.polly
	@echo "time ./matmul.normalopt"
	time -f "%E real, %U user, %S sys" ./matmul.normalopt
	@echo "time ./matmul.regular"
	time -f "%E real, %U user, %S sys" ./matmul.regular

clean:
	rm -rf  matmul.regular matmul.normalopt matmul.polly *.ll *.s
