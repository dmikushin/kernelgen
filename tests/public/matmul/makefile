LLVM = /opt/llvm
FCOMP = gfortran
DRAGONEGG_FCOMP = $(LLVM)/dragonegg/bin/dragonegg-gfortran -fplugin=$(LLVM)/dragonegg/lib64/dragonegg.so
OPT = $(LLVM)/bin/opt -load $(LLVM)/lib/LLVMPolly.so
LLC = $(LLVM)/bin/llc
#PRINT = -DPRINT

.PNONY: \
	cloog_scops_F90 cfg_scops_F90 cfg_scops_noinsts_F90 polly_scops_F90 \
	polly_deps_F90 polly_jscop_F90 polly_jscop_analyze_F90 \
	cloog_scops_c cfg_scops_c cfg_scops_noinsts_c polly_scops_c \
	polly_deps_c polly_jscop_c polly_jscop_analyze_c

all: matmul.F90.regular matmul.F90.polly matmul.c.regular matmul.c.polly

####################
#
# Fortran 90 version
#
####################

matmul.F90.regular: matmul.F90
	$(FCOMP) $(PRINT) $< -o $@

matmul.F90.llvm.s: matmul.F90
	@echo "--> 1. Create LLVM-IR from Fortran"
	$(DRAGONEGG_FCOMP) $(PRINT) -fplugin-arg-dragonegg-emit-ir -S $< -o $@

matmul.F90.preopt.ll: matmul.F90.llvm.s
	@echo "--> 2. Prepare the LLVM-IR for Polly"
	$(OPT) -S -mem2reg -loop-simplify -indvars $< > $@

cloog_scops_F90: matmul.F90.preopt.ll
	@echo "--> 3.1. Highlight SCoPs detected by Polly"
	$(OPT) -basicaa -polly-cloog -analyze -q $<

# We only create .dot files, as directly -view-scops directly calls graphviz
# which would require user interaction to continue the script.
# opt -basicaa -view-scops -disable-output matmul.F90.preopt.ll
cfg_scops_F90: matmul.F90.preopt.ll
	@echo "--> 3.2. Highlight SCoPs detected in the CFGs of the program"
	$(OPT) -basicaa -dot-scops -disable-output $<

# We only create .dot files, as directly -view-scops-only directly calls
# graphviz which would require user interaction to continue the script.
# opt -basicaa -view-scops-only -disable-output matmul.F90.preopt.ll
cfg_scops_noinsts_F90: matmul.F90.preopt.ll
	@echo "--> 3.3. Highlight SCoPs detected in the CFGs of the program (print \ no instructions)"
	$(OPT) -basicaa -dot-scops-only -disable-output $<

pngs_F90: cloog_scops_F90 cfg_scops_F90 cfg_scops_noinsts_F90
	@echo "--> 4 Create .png files from the .dot files"
	$(foreach dot, $(wildcard *.dot), $(shell dot -Tpng $(dot) > matmul.F90.$(dot).png))

polly_scops_F90: matmul.F90.preopt.ll
	@echo "--> 5. View the polyhedral representation of the SCoPs"
	$(OPT) -basicaa -polly-scops -analyze $<

polly_deps_F90: matmul.F90.preopt.ll
	@echo "--> 6. Show the dependences for the SCoPs"
	$(OPT) -basicaa -polly-dependences -analyze matmul.F90.preopt.ll

polly_jscop_F90: matmul.F90.preopt.ll
	@echo "--> 7. Export jscop files"
	$(OPT) -basicaa -polly-export-jscop matmul.F90.preopt.ll

polly_jscop_analyze_F90: polly_jscop_F90
	@echo "--> 8. Import the updated jscop files and print the new SCoPs. (optional)"
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.F90.preopt.ll
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.F90.preopt.ll \
		-polly-import-jscop-postfix=interchanged
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.F90.preopt.ll \
		-polly-import-jscop-postfix=interchanged+tiled
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.F90.preopt.ll \
		-polly-import-jscop-postfix=interchanged+tiled+vector

matmul.F90.polly.interchanged.ll: matmul.F90.preopt.ll
	@echo "--> 9.1. Codegenerate SCoPs with interchanging"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged \
		-polly-codegen $< | $(OPT) -O3 > $@

matmul.F90.polly.interchanged+tiled.ll: matmul.F90.preopt.ll
	@echo "--> 9.2. Codegenerate SCoPs with interchanging and tiling"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled \
		-polly-codegen $< | $(OPT) -O3 > $@

matmul.F90.polly.interchanged+tiled+vector.ll: matmul.F90.preopt.ll
	@echo "--> 9.3. Codegenerate SCoPs with interchanging, tiling and vectorization"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled+vector \
		-polly-codegen $< -enable-polly-vector | $(OPT) -O3 > $@

matmul.F90.polly.interchanged+tiled+vector+openmp.ll: matmul.F90.preopt.ll
	@echo "--> 9.4. Codegenerate SCoPs with interchanging, tiling and vectorization"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled+vector \
		-polly-codegen $< -enable-polly-vector -enable-polly-openmp | $(OPT) -O3 > $@

matmul.F90.normalopt.ll: matmul.F90.preopt.ll
	@echo "--> 9.5. Codegenerate with regular optimizations only"
	$(OPT) $< | $(OPT) -O3 > $@

matmul.F90.polly.interchanged.s: matmul.F90.polly.interchanged.ll
	@echo "--> 10.1. Emit asm for SCoPs with interchanging"
	$(LLC) $< -o $@

matmul.F90.polly.interchanged+tiled.s: matmul.F90.polly.interchanged+tiled.ll
	@echo "--> 10.2. Emit asm for SCoPs with interchanging and tiling"
	$(LLC) $< -o $@

matmul.F90.polly.interchanged+tiled+vector.s: matmul.F90.polly.interchanged+tiled+vector.ll
	@echo "--> 10.3. Emit asm for SCoPs with interchanging, tiling and vectorization"
	$(LLC) $< -o $@

matmul.F90.polly.interchanged+tiled+vector+openmp.s: matmul.F90.polly.interchanged+tiled+vector+openmp.ll
	@echo "--> 10.4. Emit asm for SCoPs with interchanging, tiling and vectorization"
	$(LLC) $< -o $@

matmul.F90.normalopt.s: matmul.F90.normalopt.ll
	@echo "--> 10.5. Emit asm for code with regular optimizations only"
	$(LLC) $< -o $@

matmul.F90.polly.interchanged: matmul.F90.polly.interchanged.s
	@echo "--> 11.1. Create executables for SCoPs with interchanging"
	$(FCOMP) $< -o $@

matmul.F90.polly.interchanged+tiled: matmul.F90.polly.interchanged+tiled.s
	@echo "--> 11.2. Create executables for SCoPs with interchanging and tiling"
	$(FCOMP) $< -o $@

matmul.F90.polly.interchanged+tiled+vector: matmul.F90.polly.interchanged+tiled+vector.s
	@echo "--> 11.3. Create executables for SCoPs with interchanging, tiling and vectorization"
	$(FCOMP) $< -o $@

matmul.F90.polly.interchanged+tiled+vector+openmp: matmul.F90.polly.interchanged+tiled+vector+openmp.s
	@echo "--> 11.4. Create executables for SCoPs with interchanging, tiling and vectorization"
	$(FCOMP) -lgomp $< -o $@

matmul.F90.normalopt: matmul.F90.normalopt.s
	@echo "--> 11.5. Create executables for code with regular optimizations only"
	$(FCOMP) $< -o $@

matmul.F90.polly: pngs_F90 polly_scops_F90 polly_deps_F90 polly_jscop_F90 polly_jscop_analyze_F90 \
	matmul.F90.normalopt matmul.F90.polly.interchanged matmul.F90.polly.interchanged+tiled \
	matmul.F90.polly.interchanged+tiled+vector matmul.F90.polly.interchanged+tiled+vector+openmp

benchmark_F90: matmul.F90.polly
	@echo "--> 12. Compare the runtime of the executables"
	@echo "time ./matmul.F90.normalopt.exe"
	time -f "%E real, %U user, %S sys" ./matmul.F90.normalopt
	@echo "time ./matmul.F90.polly.interchanged"
	time -f "%E real, %U user, %S sys" ./matmul.F90.polly.interchanged
	@echo "time ./matmul.F90.polly.interchanged+tiled"
	time -f "%E real, %U user, %S sys" ./matmul.F90.polly.interchanged+tiled
	@echo "time ./matmul.F90.polly.interchanged+tiled+vector"
	time -f "%E real, %U user, %S sys" ./matmul.F90.polly.interchanged+tiled+vector
	@echo "time ./matmul.F90.polly.interchanged+tiled+vector+openmp"
	time -f "%E real, %U user, %S sys" ./matmul.F90.polly.interchanged+tiled+vector+openmp

####################
#
# C version
#
####################

matmul.c.regular: matmul.c
	$(FCOMP) $(PRINT) $< -o $@

matmul.c.llvm.s: matmul.c
	@echo "--> 1. Create LLVM-IR from Fortran"
	$(DRAGONEGG_FCOMP) $(PRINT) -fplugin-arg-dragonegg-emit-ir -S $< -o $@

matmul.c.preopt.ll: matmul.c.llvm.s
	@echo "--> 2. Prepare the LLVM-IR for Polly"
	$(OPT) -S -mem2reg -loop-simplify -indvars $< > $@

cloog_scops_c: matmul.c.preopt.ll
	@echo "--> 3.1. Highlight SCoPs detected by Polly"
	$(OPT) -basicaa -polly-cloog -analyze -q $<

# We only create .dot files, as directly -view-scops directly calls graphviz
# which would require user interaction to continue the script.
# opt -basicaa -view-scops -disable-output matmul.c.preopt.ll
cfg_scops_c: matmul.c.preopt.ll
	@echo "--> 3.2. Highlight SCoPs detected in the CFGs of the program"
	$(OPT) -basicaa -dot-scops -disable-output $<

# We only create .dot files, as directly -view-scops-only directly calls
# graphviz which would require user interaction to continue the script.
# opt -basicaa -view-scops-only -disable-output matmul.c.preopt.ll
cfg_scops_noinsts_c: matmul.c.preopt.ll
	@echo "--> 3.3. Highlight SCoPs detected in the CFGs of the program (print no instructions)"
	$(OPT) -basicaa -dot-scops-only -disable-output $<

pngs_c: cloog_scops_c cfg_scops_c cfg_scops_noinsts_c
	@echo "--> 4 Create .png files from the .dot files"
	$(foreach dot, $(wildcard *.dot), $(shell dot -Tpng $(dot) > matmul.c.$(dot).png))

polly_scops_c: matmul.c.preopt.ll
	@echo "--> 5. View the polyhedral representation of the SCoPs"
	$(OPT) -basicaa -polly-scops -analyze $<

polly_deps_c: matmul.c.preopt.ll
	@echo "--> 6. Show the dependences for the SCoPs"
	$(OPT) -basicaa -polly-dependences -analyze matmul.c.preopt.ll

polly_jscop_c: matmul.c.preopt.ll
	@echo "--> 7. Export jscop files"
	$(OPT) -basicaa -polly-export-jscop matmul.c.preopt.ll

polly_jscop_analyze_c: polly_jscop_c
	@echo "--> 8. Import the updated jscop files and print the new SCoPs. (optional)"
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.c.preopt.ll
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.c.preopt.ll \
		-polly-import-jscop-postfix=interchanged
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.c.preopt.ll \
		-polly-import-jscop-postfix=interchanged+tiled
	$(OPT) -basicaa -polly-import-jscop -polly-cloog -analyze matmul.c.preopt.ll \
		-polly-import-jscop-postfix=interchanged+tiled+vector

matmul.c.polly.interchanged.ll: matmul.c.preopt.ll
	@echo "--> 9.1. Codegenerate SCoPs with interchanging"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged \
		-polly-codegen $< | $(OPT) -O3 > $@

matmul.c.polly.interchanged+tiled.ll: matmul.c.preopt.ll
	@echo "--> 9.2. Codegenerate SCoPs with interchanging and tiling"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled \
		-polly-codegen $< | $(OPT) -O3 > $@

matmul.c.polly.interchanged+tiled+vector.ll: matmul.c.preopt.ll
	@echo "--> 9.3. Codegenerate SCoPs with interchanging, tiling and vectorization"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled+vector \
		-polly-codegen $< -enable-polly-vector | $(OPT) -O3 > $@

matmul.c.polly.interchanged+tiled+vector+openmp.ll: matmul.c.preopt.ll
	@echo "--> 9.4. Codegenerate SCoPs with interchanging, tiling and vectorization"
	$(OPT) -basicaa -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled+vector \
		-polly-codegen $< -enable-polly-vector -enable-polly-openmp | $(OPT) -O3 > $@

matmul.c.normalopt.ll: matmul.c.preopt.ll
	@echo "--> 9.5. Codegenerate with regular optimizations only"
	$(OPT) $< | $(OPT) -O3 > $@

matmul.c.polly.interchanged.s: matmul.c.polly.interchanged.ll
	@echo "--> 10.1. Emit asm for SCoPs with interchanging"
	$(LLC) $< -o $@

matmul.c.polly.interchanged+tiled.s: matmul.c.polly.interchanged+tiled.ll
	@echo "--> 10.2. Emit asm for SCoPs with interchanging and tiling"
	$(LLC) $< -o $@

matmul.c.polly.interchanged+tiled+vector.s: matmul.c.polly.interchanged+tiled+vector.ll
	@echo "--> 10.3. Emit asm for SCoPs with interchanging, tiling and vectorization"
	$(LLC) $< -o $@

matmul.c.polly.interchanged+tiled+vector+openmp.s: matmul.c.polly.interchanged+tiled+vector+openmp.ll
	@echo "--> 10.4. Emit asm for SCoPs with interchanging, tiling and vectorization"
	$(LLC) $< -o $@

matmul.c.normalopt.s: matmul.c.normalopt.ll
	@echo "--> 10.5. Emit asm for code with regular optimizations only"
	$(LLC) $< -o $@

matmul.c.polly.interchanged: matmul.c.polly.interchanged.s
	@echo "--> 11.1. Create executables for SCoPs with interchanging"
	$(FCOMP) $< -o $@

matmul.c.polly.interchanged+tiled: matmul.c.polly.interchanged+tiled.s
	@echo "--> 11.2. Create executables for SCoPs with interchanging and tiling"
	$(FCOMP) $< -o $@

matmul.c.polly.interchanged+tiled+vector: matmul.c.polly.interchanged+tiled+vector.s
	@echo "--> 11.3. Create executables for SCoPs with interchanging, tiling and vectorization"
	$(FCOMP) $< -o $@

matmul.c.polly.interchanged+tiled+vector+openmp: matmul.c.polly.interchanged+tiled+vector+openmp.s
	@echo "--> 11.4. Create executables for SCoPs with interchanging, tiling and vectorization"
	$(FCOMP) -lgomp $< -o $@

matmul.c.normalopt: matmul.c.normalopt.s
	@echo "--> 11.5. Create executables for code with regular optimizations only"
	$(FCOMP) $< -o $@

matmul.c.polly: pngs_c polly_scops_c polly_deps_c polly_jscop_c polly_jscop_analyze_c \
	matmul.c.normalopt matmul.c.polly.interchanged matmul.c.polly.interchanged+tiled \
	matmul.c.polly.interchanged+tiled+vector matmul.c.polly.interchanged+tiled+vector+openmp

benchmark_c: matmul.c.polly
	@echo "--> 12. Compare the runtime of the executables"
	@echo "time ./matmul.c.normalopt.exe"
	time -f "%E real, %U user, %S sys" ./matmul.c.normalopt
	@echo "time ./matmul.c.polly.interchanged"
	time -f "%E real, %U user, %S sys" ./matmul.c.polly.interchanged
	@echo "time ./matmul.c.polly.interchanged+tiled"
	time -f "%E real, %U user, %S sys" ./matmul.c.polly.interchanged+tiled
	@echo "time ./matmul.c.polly.interchanged+tiled+vector"
	time -f "%E real, %U user, %S sys" ./matmul.c.polly.interchanged+tiled+vector
	@echo "time ./matmul.c.polly.interchanged+tiled+vector+openmp"
	time -f "%E real, %U user, %S sys" ./matmul.c.polly.interchanged+tiled+vector+openmp

clean:
	rm -rf  matmul.F90.regular *.ll *.s *.dot *.jscop *.png \
		matmul.F90.normalopt matmul.F90.polly.interchanged matmul.F90.polly.interchanged+tiled \
		matmul.F90.polly.interchanged+tiled+vector matmul.F90.polly.interchanged+tiled+vector+openmp
	rm -rf  matmul.c.regular *.ll *.s *.dot *.png \
		matmul.c.normalopt matmul.c.polly.interchanged matmul.c.polly.interchanged+tiled \
		matmul.c.polly.interchanged+tiled+vector matmul.c.polly.interchanged+tiled+vector+openmp
