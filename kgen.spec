AutoReq: 0

Name:           kernelgen
Version:        0.1
Release:        alpha
Summary:        Compiler with automatic generation of GPU kernels from Fortran source code 
Source0:	ftp://upload.hpcforge.org/pub/kernelgen/llvm-r136600.tar.gz
Source1:	ftp://upload.hpcforge.org/pub/kernelgen/gcc-4.5-r177629.tar.gz
Source2:	ftp://upload.hpcforge.org/pub/kernelgen/dragonegg-r136347.tar.gz
Source3:	ftp://upload.hpcforge.org/pub/kernelgen/kernelgen-r375.tar.gz
Source4:	ftp://upload.hpcforge.org/pub/kernelgen/polly-r137304.tar.gz
Source5:	ftp://upload.hpcforge.org/pub/kernelgen/cloog-225c2ed62fe37a4db22bf4b95c3731dab1a50dde.tar.gz
Source6:	ftp://upload.hpcforge.org/pub/kernelgen/scoplib-0.2.0.tar.gz
Patch0:		llvm.varargs.patch
Patch1:		llvm.patch
Patch2:		llvm.gpu.patch
Patch3:		gcc.patch
Patch4:		gcc.opencl.patch
Patch5:		dragonegg.opencl.patch

Group:          Applications/Engineering
License:        GPL/BSD/Freeware
URL:            https://hpcforge.org/projects/kernelgen/

BuildRequires:  gcc-gfortran perl elfutils-libelf-devel libffi-devel gmp-devel mpfr-devel libmpc-devel flex glibc-devel
Requires:       gcc-gfortran perl elfutils-libelf libffi gmp mpfr libmpc perl-IPC-Run3 perl-XML-LibXSLT

Packager:       Dmitry Mikushin <dmikushin@nvidia.com>

%description
KGen is a tool for automatic generation of GPU kernels from Fortran source code. From user's point of view it acts as regular Fortran compiler.


%prep
rm -rf $RPM_BUILD_DIR/llvm
tar -xf llvm-r136600.tar.gz
cd $RPM_BUILD_DIR/llvm/tools
tar -xf ../../polly-r137304.tar.gz
cd $RPM_BUILD_DIR
rm -rf $RPM_BUILD_DIR/gcc-4.5
tar -xf gcc-4.5-r177629.tar.gz
rm -rf $RPM_BUILD_DIR/dragonegg
tar -xf dragonegg-r136347.tar.gz
rm -rf $RPM_BUILD_DIR/kernelgen
tar -xf kernelgen-r375.tar.gz
rm -rf $RPM_BUILD_DIR/cloog
tar -xf cloog-225c2ed62fe37a4db22bf4b95c3731dab1a50dde.tar.gz
rm -rf $RPM_BUILD_DIR/scoplib-0.2.0
tar -xf scoplib-0.2.0.tar.gz

%patch0 -p1
%patch1 -p1
%patch2 -p1
%patch3 -p1
%patch4 -p1
%patch5 -p1


%build
cd $RPM_BUILD_DIR/cloog
./get_submodules.sh
./autogen.sh
./configure --prefix=$RPM_BUILD_ROOT/opt/kgen
make
ln -s $RPM_BUILD_DIR/cloog/.libs $RPM_BUILD_DIR/cloog/lib
ln -s $RPM_BUILD_DIR/cloog/isl/.libs $RPM_BUILD_DIR/cloog/isl/lib
cd $RPM_BUILD_DIR/scoplib-0.2.0
./configure --enable-mp-version --prefix=$RPM_BUILD_ROOT/opt/kgen
make
ln -s $RPM_BUILD_DIR/scoplib-0.2.0/source/.libs $RPM_BUILD_DIR/scoplib-0.2.0/lib
cd $RPM_BUILD_DIR/llvm
mkdir build
cp -rf include/ build/include/
cd build
../configure --enable-jit --enable-optimized --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kgen --enable-targets=host,cbe --with-cloog=$RPM_BUILD_DIR/cloog --with-isl=$RPM_BUILD_DIR/cloog/isl --with-scoplib=$RPM_BUILD_DIR/scoplib-0.2.0
make -j12 CXXFLAGS=-O3
cd $RPM_BUILD_DIR/gcc-4.5
mkdir build
cd build/
../configure --prefix=$RPM_BUILD_ROOT/opt/kgen --program-prefix=dragonegg- --enable-languages=fortran --with-mpfr-include=/usr/include/ --with-mpfr-lib=/usr/lib64 --with-gmp-include=/usr/include/ --with-gmp-lib=/usr/lib64 --enable-plugin
make -j12
cd $RPM_BUILD_DIR/kernelgen/trunk
./configure
make src


%install
cd $RPM_BUILD_DIR/cloog
make install
cd $RPM_BUILD_DIR/scoplib-0.2.0
make install
cd $RPM_BUILD_DIR/llvm/build
make install
cd $RPM_BUILD_DIR/gcc-4.5/build
make install
cd $RPM_BUILD_DIR/dragonegg
GCC=$RPM_BUILD_ROOT/opt/kgen/bin/dragonegg-gcc LLVM_CONFIG=$RPM_BUILD_ROOT/opt/kgen/bin/llvm-config make
cp dragonegg.so $RPM_BUILD_ROOT/opt/kgen/lib64/
cd $RPM_BUILD_DIR/kernelgen/trunk
PREFIX=$RPM_BUILD_ROOT make install
rm $RPM_BUILD_ROOT/opt/kgen/bin/bugpoint
rm $RPM_BUILD_ROOT/opt/kgen/bin/dragonegg-cpp
rm $RPM_BUILD_ROOT/opt/kgen/bin/dragonegg-gcc
rm $RPM_BUILD_ROOT/opt/kgen/bin/dragonegg-gccbug
rm $RPM_BUILD_ROOT/opt/kgen/bin/dragonegg-gcov
rm $RPM_BUILD_ROOT/opt/kgen/bin/lli
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-ar
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-as
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-bcanalyzer
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-config
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-diff
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-dis
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-extract
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-ld
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-link
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-mc
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-nm
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-objdump
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-prof
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-ranlib
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-rtdyld
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvm-stub
rm $RPM_BUILD_ROOT/opt/kgen/bin/llvmc
rm $RPM_BUILD_ROOT/opt/kgen/bin/macho-dump
rm $RPM_BUILD_ROOT/opt/kgen/bin/tblgen
rm $RPM_BUILD_ROOT/opt/kgen/bin/x86_64-unknown-linux-gnu-dragonegg-gcc
rm $RPM_BUILD_ROOT/opt/kgen/bin/x86_64-unknown-linux-gnu-dragonegg-gfortran
rm $RPM_BUILD_ROOT/opt/kgen/bin/x86_64-unknown-linux-gnu-gcc-4.5.4
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html.tar.gz
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/AliasAnalysis.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/BitCodeFormat.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/BranchWeightMetadata.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/Bugpoint.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CFEBuildInstrs.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CMake.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CodeGenerator.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CodingStandards.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/FileCheck.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/bugpoint.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/index.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/lit.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llc.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/lli.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-ar.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-as.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-bcanalyzer.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-config.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-diff.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-dis.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-extract.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-ld.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-link.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-nm.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-prof.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvm-ranlib.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvmc.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvmgcc.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/llvmgxx.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/manpage.css
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/opt.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandGuide/tblgen.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CommandLine.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CompilerDriver.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CompilerDriverTutorial.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/CompilerWriterInfo.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/DebuggingJITedCode.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/DeveloperPolicy.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/ExceptionHandling.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/ExtendingLLVM.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/FAQ.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/GCCFEBuildInstrs.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/GarbageCollection.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/GetElementPtr.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/GettingStarted.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/GettingStartedVS.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/GoldPlugin.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/HowToReleaseLLVM.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/HowToSubmitABug.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/LangRef.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/Lexicon.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/LinkTimeOptimization.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/MakefileGuide.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/Packaging.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/Passes.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/ProgrammersManual.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/Projects.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/ReleaseNotes.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/SourceLevelDebugging.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/SystemLibrary.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/TableGenFundamentals.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/TestingGuide.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/UsingLibraries.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/WritingAnLLVMBackend.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/WritingAnLLVMPass.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/doxygen.css
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/img/Debugging.gif
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/img/libdeps.gif
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/img/lines.gif
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/img/objdeps.gif
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/img/venusflytrap.jpg
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/index.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/llvm.css
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl1.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl2.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl3.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl4.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl5.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl6.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl7.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/LangImpl8.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl1.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl2.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl3.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl4.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl5.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl6.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl7.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/OCamlLangImpl8.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/html/tutorial/index.html
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/FileCheck.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/bugpoint.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/lit.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llc.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/lli.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-ar.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-as.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-bcanalyzer.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-config.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-diff.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-dis.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-extract.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-ld.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-link.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-nm.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-prof.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvm-ranlib.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvmc.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvmgcc.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/llvmgxx.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/opt.ps
rm $RPM_BUILD_ROOT/opt/kgen/docs/llvm/ps/tblgen.ps
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/block.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/clast.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/cloog.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/constraints.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/domain.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/input.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/int.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/isl/backend.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/isl/cloog.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/isl/constraintset.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/isl/domain.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/loop.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/matrix.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/matrix/constraintset.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/names.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/options.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/pprint.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/program.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/state.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/statement.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/stride.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/union_domain.h
rm $RPM_BUILD_ROOT/opt/kgen/include/cloog/version.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/aff.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/aff_type.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/arg.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/band.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/blk.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/config.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/constraint.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/ctx.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/dim.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/div.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/flow.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/hash.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/ilp.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/int.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/list.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/local_space.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/lp.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/map.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/map_type.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/mat.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/obj.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/options.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/point.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/polynomial.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/printer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/schedule.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/seq.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/set.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/set_type.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/stdint.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/stream.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/union_map.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/union_set.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/vec.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/version.h
rm $RPM_BUILD_ROOT/opt/kgen/include/isl/vertices.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Analysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/BitReader.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/BitWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Core.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Disassembler.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/EnhancedDisassembly.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/ExecutionEngine.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Initialization.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/LinkTimeOptimizer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Object.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Target.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Transforms/IPO.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/Transforms/Scalar.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm-c/lto.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/APFloat.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/APInt.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/APSInt.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ArrayRef.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/BitVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/DAGDeltaAlgorithm.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/DeltaAlgorithm.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/DenseMap.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/DenseMapInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/DenseSet.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/DepthFirstIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/EquivalenceClasses.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/FoldingSet.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/GraphTraits.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ImmutableIntervalMap.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ImmutableList.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ImmutableMap.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ImmutableSet.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/InMemoryStruct.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/IndexedMap.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/IntEqClasses.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/IntervalMap.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/IntrusiveRefCntPtr.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/NullablePtr.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/Optional.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/OwningPtr.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/PackedVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/PointerIntPair.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/PointerUnion.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/PostOrderIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/PriorityQueue.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SCCIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/STLExtras.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ScopedHashTable.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SetOperations.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SetVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SmallBitVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SmallPtrSet.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SmallSet.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SmallString.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SmallVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/SparseBitVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/Statistic.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/StringExtras.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/StringMap.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/StringRef.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/StringSet.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/StringSwitch.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/TinyPtrVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/Trie.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/Triple.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/Twine.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/UniqueVector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ValueMap.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/VectorExtras.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ilist.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ADT/ilist_node.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/AliasAnalysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/AliasSetTracker.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/BlockFrequencyImpl.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/BlockFrequencyInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/BranchProbabilityInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/CFGPrinter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/CallGraph.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/CaptureTracking.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/CodeMetrics.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ConstantFolding.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ConstantsScanner.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/DIBuilder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/DOTGraphTraitsPass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/DebugInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/DomPrinter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/DominanceFrontier.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/DominatorInternals.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/Dominators.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/FindUsedTypes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/IVUsers.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/InlineCost.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/InstructionSimplify.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/Interval.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/IntervalIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/IntervalPartition.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/LazyValueInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/LibCallAliasAnalysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/LibCallSemantics.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/Lint.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/Loads.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/LoopDependenceAnalysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/LoopInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/LoopPass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/MemoryBuiltins.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/MemoryDependenceAnalysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/PHITransAddr.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/Passes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/PathNumbering.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/PathProfileInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/PostDominators.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ProfileInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ProfileInfoLoader.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ProfileInfoTypes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/RegionInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/RegionIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/RegionPass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/RegionPrinter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ScalarEvolution.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ScalarEvolutionExpander.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ScalarEvolutionExpressions.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ScalarEvolutionNormalization.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/SparsePropagation.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/Trace.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/ValueTracking.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Analysis/Verifier.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Argument.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Assembly/AssemblyAnnotationWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Assembly/Parser.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Assembly/PrintModulePass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Assembly/Writer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Attributes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/AutoUpgrade.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/BasicBlock.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Bitcode/Archive.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Bitcode/BitCodes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Bitcode/BitstreamReader.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Bitcode/BitstreamWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Bitcode/LLVMBitCodes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Bitcode/ReaderWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CallGraphSCCPass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CallingConv.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/Analysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/AsmPrinter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/BinaryObject.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/CalcSpillWeights.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/CallingConvLower.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/EdgeBundles.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/FastISel.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/FunctionLoweringInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/GCMetadata.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/GCMetadataPrinter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/GCStrategy.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/GCs.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ISDOpcodes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/IntrinsicLowering.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/JITCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/LatencyPriorityQueue.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/LinkAllAsmWriterComponents.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/LinkAllCodegenComponents.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/LiveInterval.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/LiveIntervalAnalysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/LiveStackAnalysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/LiveVariables.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachORelocation.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineBasicBlock.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineBlockFrequencyInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineBranchProbabilityInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineCodeInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineConstantPool.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineDominators.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineFrameInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineFunction.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineFunctionAnalysis.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineFunctionPass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineInstr.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineInstrBuilder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineJumpTableInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineLoopInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineLoopRanges.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineMemOperand.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineModuleInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineModuleInfoImpls.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineOperand.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachinePassRegistry.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineRegisterInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineRelocation.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/MachineSSAUpdater.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ObjectCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/PBQP/Graph.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/PBQP/HeuristicBase.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/PBQP/HeuristicSolver.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/PBQP/Heuristics/Briggs.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/PBQP/Math.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/PBQP/Solution.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/Passes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ProcessImplicitDefs.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/PseudoSourceValue.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/RegAllocPBQP.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/RegAllocRegistry.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/RegisterScavenging.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/RuntimeLibcalls.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ScheduleDAG.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ScheduleHazardRecognizer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/SchedulerRegistry.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ScoreboardHazardRecognizer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/SelectionDAG.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/SelectionDAGISel.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/SelectionDAGNodes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/SlotIndexes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/TargetLoweringObjectFileImpl.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ValueTypes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CodeGen/ValueTypes.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/Action.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/AutoGenerated.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/BuiltinOptions.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/Common.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/CompilationGraph.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/Error.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/Main.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/Main.inc
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/CompilerDriver/Tool.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Config/AsmParsers.def
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Config/AsmPrinters.def
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Config/Disassemblers.def
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Config/Targets.def
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Config/config.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Config/llvm-config.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Constant.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Constants.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/DebugInfoProbe.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/DefaultPasses.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/DerivedTypes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/ExecutionEngine.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/GenericValue.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/Interpreter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/JIT.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/JITEventListener.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/JITMemoryManager.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/MCJIT.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ExecutionEngine/RuntimeDyld.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Function.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/GVMaterializer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/GlobalAlias.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/GlobalValue.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/GlobalVariable.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/InitializePasses.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/InlineAsm.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/InstrTypes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Instruction.def
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Instruction.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Instructions.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicInst.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Intrinsics.gen
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Intrinsics.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Intrinsics.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicsARM.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicsAlpha.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicsCellSPU.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicsPTX.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicsPowerPC.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicsX86.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/IntrinsicsXCore.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/LLVMContext.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/LinkAllPasses.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/LinkAllVMCore.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Linker.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/EDInstInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCAsmBackend.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCAsmInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCAsmInfoCOFF.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCAsmInfoDarwin.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCAsmLayout.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCAssembler.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCCodeGenInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCContext.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCDirectives.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCDisassembler.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCDwarf.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCELFObjectWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCELFSymbolFlags.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCExpr.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCFixup.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCFixupKindInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCInst.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCInstPrinter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCInstrDesc.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCInstrInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCInstrItineraries.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCLabel.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCMachOSymbolFlags.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCMachObjectWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCObjectFileInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCObjectStreamer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCObjectWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCParser/AsmCond.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCParser/AsmLexer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCParser/MCAsmLexer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCParser/MCAsmParser.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCParser/MCAsmParserExtension.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCParser/MCParsedAsmOperand.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCRegisterInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCSection.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCSectionCOFF.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCSectionELF.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCSectionMachO.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCStreamer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCSubtargetInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCSymbol.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCTargetAsmLexer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCTargetAsmParser.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCValue.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MCWin64EH.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/MachineLocation.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/SectionKind.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/MC/SubtargetFeature.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Metadata.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Module.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Object/Binary.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Object/COFF.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Object/Error.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Object/MachOFormat.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Object/MachOObject.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Object/ObjectFile.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/OperandTraits.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Operator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Pass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/PassAnalysisSupport.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/PassManager.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/PassManagers.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/PassRegistry.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/PassSupport.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/AIXDataTypesFix.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/AlignOf.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Allocator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Atomic.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/BlockFrequency.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/BranchProbability.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/CFG.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/COFF.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/CallSite.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Capacity.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Casting.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/CommandLine.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Compiler.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ConstantFolder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ConstantRange.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/CrashRecoveryContext.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/DOTGraphTraits.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/DataFlow.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/DataTypes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Debug.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/DebugLoc.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Disassembler.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Dwarf.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/DynamicLibrary.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ELF.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Endian.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Errno.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ErrorHandling.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/FEnv.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/FileSystem.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/FileUtilities.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Format.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/FormattedStream.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/GetElementPtrTypeIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/GraphWriter.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Host.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/IRBuilder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/IRReader.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/IncludeFile.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/InstIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/InstVisitor.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/LICENSE.TXT
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/LeakDetector.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/MachO.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ManagedStatic.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/MathExtras.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Memory.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/MemoryBuffer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/MemoryObject.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Mutex.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/MutexGuard.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/NoFolder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/OutputBuffer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PassManagerBuilder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PassNameParser.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Path.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PathV1.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PathV2.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PatternMatch.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PluginLoader.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PointerLikeTypeTraits.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PredIteratorCache.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/PrettyStackTrace.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Process.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Program.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/RWMutex.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Recycler.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/RecyclingAllocator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Regex.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Registry.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/RegistryParser.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/SMLoc.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Signals.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Solaris.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/SourceMgr.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/StringPool.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/SwapByteOrder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/SystemUtils.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/TargetFolder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ThreadLocal.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Threading.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/TimeValue.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Timer.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ToolOutputFile.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/TypeBuilder.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Valgrind.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/ValueHandle.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/Win64EH.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/circular_raw_ostream.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/raw_os_ostream.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/raw_ostream.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/system_error.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Support/type_traits.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/SymbolTableListTraits.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/Mangler.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/Target.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetCallingConv.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetCallingConv.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetData.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetELFWriterInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetFrameLowering.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetInstrInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetIntrinsicInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetJITInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetLibraryInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetLowering.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetLoweringObjectFile.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetMachine.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetOpcodes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetOptions.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetRegisterInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetRegistry.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetSchedule.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetSelect.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetSelectionDAG.td
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetSelectionDAGInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Target/TargetSubtargetInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/IPO.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/IPO/InlinerPass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Instrumentation.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Scalar.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/AddrModeMatcher.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/BasicBlockUtils.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/BasicInliner.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/BuildLibCalls.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/Cloning.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/FunctionUtils.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/Local.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/PromoteMemToReg.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/SSAUpdater.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/SSAUpdaterImpl.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/UnifyFunctionExitNodes.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/UnrollLoop.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Transforms/Utils/ValueMapper.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Type.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Use.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/User.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/Value.h
rm $RPM_BUILD_ROOT/opt/kgen/include/llvm/ValueSymbolTable.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/Cloog.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/Config/config.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/Dependences.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/LinkAllPasses.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/MayAliasSet.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/ScopDetection.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/ScopInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/ScopLib.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/ScopPass.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/Support/AffineSCEVIterator.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/Support/GICHelper.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/Support/ScopHelper.h
rm $RPM_BUILD_ROOT/opt/kgen/include/polly/TempScopInfo.h
rm $RPM_BUILD_ROOT/opt/kgen/include/scoplib/macros.h
rm $RPM_BUILD_ROOT/opt/kgen/include/scoplib/matrix.h
rm $RPM_BUILD_ROOT/opt/kgen/include/scoplib/scop.h
rm $RPM_BUILD_ROOT/opt/kgen/include/scoplib/statement.h
rm $RPM_BUILD_ROOT/opt/kgen/include/scoplib/vector.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/BugpointPasses.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/LLVMHello.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtbegin.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtbeginS.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtbeginT.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtend.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtendS.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtfastmath.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtprec32.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtprec64.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/crtprec80.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/libgcc.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/libgcc_eh.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/libgcov.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/libgfortranbegin.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/32/libgfortranbegin.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtbegin.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtbeginS.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtbeginT.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtend.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtendS.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtfastmath.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtprec32.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtprec64.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/crtprec80.o
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/finclude/omp_lib.dragonegg.mod
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/finclude/omp_lib.f90
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/finclude/omp_lib.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/finclude/omp_lib_kinds.dragonegg.mod
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include-fixed/README
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include-fixed/limits.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include-fixed/linux/a.out.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include-fixed/syslimits.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/abmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/ammintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/avxintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/bmmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/cpuid.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/cross-stdarg.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/emmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/float.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/fma4intrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/ia32intrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/immintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/iso646.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/lwpintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/mf-runtime.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/mm3dnow.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/mm_malloc.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/mmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/nmmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/omp.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/pmmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/popcntintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/smmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/ssp/ssp.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/ssp/stdio.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/ssp/string.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/ssp/unistd.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/stdarg.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/stdbool.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/stddef.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/stdfix.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/stdint-gcc.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/stdint.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/tmmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/unwind.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/varargs.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/wmmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/x86intrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/xmmintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/include/xopintrin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/fixinc_list
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/gsyslimits.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/include/README
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/include/limits.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/macro_list
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/mkheaders.conf
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/libgcc.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/libgcc_eh.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/libgcov.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/libgfortranbegin.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/libgfortranbegin.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/ada/gcc-interface/ada-tree.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/alias.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/all-tree.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/ansidecl.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/auto-host.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/b-header-vars
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/basic-block.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/bitmap.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/builtins.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/bversion.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/c-common.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/c-common.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/c-pragma.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/c-pretty-print.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/cfghooks.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/cfgloop.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/cgraph.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/cif-code.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/dbxelf.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/elfos.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/glibc-stdint.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/i386/att.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/i386/biarch64.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/i386/i386-protos.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/i386/i386.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/i386/linux64.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/i386/unix.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/i386/x86-64.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/linux.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/svr4.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/config/vxworks-dummy.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/configargs.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/coretypes.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/cp/cp-tree.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/cppdefault.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/cpplib.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/debug.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/defaults.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/diagnostic.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/diagnostic.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/double-int.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/emit-rtl.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/except.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/filenames.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/fixed-value.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/flags.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/function.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/gcc-plugin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/genrtl.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/ggc.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/gimple.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/gimple.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/gsstruct.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/gtype-desc.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/hard-reg-set.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/hashtab.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/highlev-plugin-common.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/hwint.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/incpath.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/input.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/insn-constants.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/insn-flags.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/insn-modes.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/insn-notes.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/intl.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/ipa-prop.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/ipa-reference.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/ipa-utils.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/java/java-tree.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/langhooks.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/libiberty.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/line-map.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/machmode.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/md5.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/mode-classes.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/objc/objc-tree.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/obstack.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/omp-builtins.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/options.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/opts.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/output.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/params.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/params.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/partition.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/plugin-version.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/plugin.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/plugin.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/pointer-set.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/predict.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/predict.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/prefix.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/pretty-print.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/real.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/reg-notes.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/rtl.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/rtl.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/safe-ctype.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/sbitmap.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/splay-tree.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/statistics.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/symtab.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/sync-builtins.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/system.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/target.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/timevar.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/timevar.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tm-preds.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tm.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tm_p.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/toplev.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-check.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-dump.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-flow-inline.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-flow.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-inline.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-iterator.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-pass.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-ssa-alias.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-ssa-operands.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree-ssa-sccvn.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/tree.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/treestruct.def
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/varray.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/vec.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/vecprim.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/gcc/x86_64-unknown-linux-gnu/4.5.4/plugin/include/version.h
rm $RPM_BUILD_ROOT/opt/kgen/lib/libCompilerDriver.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libEnhancedDisassembly.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libEnhancedDisassembly.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMAnalysis.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMArchive.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMAsmParser.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMAsmPrinter.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMBitReader.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMBitWriter.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMCBackend.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMCBackendInfo.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMCodeGen.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMCore.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMExecutionEngine.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMInstCombine.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMInstrumentation.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMInterpreter.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMJIT.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMLinker.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMMC.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMMCDisassembler.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMMCJIT.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMMCParser.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMObject.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMRuntimeDyld.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMScalarOpts.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMSelectionDAG.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMSupport.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMTarget.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMTransformUtils.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMX86AsmParser.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMX86AsmPrinter.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMX86CodeGen.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMX86Desc.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMX86Disassembler.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMX86Info.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMX86Utils.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMipa.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLLVMipo.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLTO.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libLTO.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libcloog-isl.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libcloog-isl.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libcloog-isl.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libcloog-isl.so.2
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgcc_s.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgcc_s.so.1
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgfortran.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgfortran.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgfortran.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgfortran.so.3
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgfortran.so.3.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgomp.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgomp.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgomp.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgomp.so.1
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgomp.so.1.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libgomp.spec
rm $RPM_BUILD_ROOT/opt/kgen/lib/libisl.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libisl.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libisl.so.7.0.0-gdb.py
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflap.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflap.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflap.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflap.so.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflap.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflapth.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflapth.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflapth.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflapth.so.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libmudflapth.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libpollyanalysis.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libpollyexchange.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libpollyjson.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libpollysupport.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libprofile_rt.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libprofile_rt.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libscoplib.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libscoplib.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libscoplib.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libscoplib.so.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libssp.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libssp.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/libssp.so
rm $RPM_BUILD_ROOT/opt/kgen/lib/libssp.so.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libssp.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib/libssp_nonshared.a
rm $RPM_BUILD_ROOT/opt/kgen/lib/libssp_nonshared.la
rm $RPM_BUILD_ROOT/opt/kgen/lib/pkgconfig/cloog-isl.pc
rm $RPM_BUILD_ROOT/opt/kgen/lib/pkgconfig/isl.pc
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgcc_s.so
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgcc_s.so.1
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgfortran.a
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgfortran.la
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgfortran.so
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgfortran.so.3
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgfortran.so.3.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgomp.a
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgomp.la
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgomp.so
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgomp.so.1
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgomp.so.1.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libgomp.spec
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libiberty.a
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflap.a
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflap.la
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflap.so
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflap.so.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflap.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflapth.a
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflapth.la
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflapth.so
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflapth.so.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libmudflapth.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libssp.a
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libssp.la
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libssp.so
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libssp.so.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libssp.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libssp_nonshared.a
rm $RPM_BUILD_ROOT/opt/kgen/lib64/libssp_nonshared.la
rm $RPM_BUILD_ROOT/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/cc1
rm $RPM_BUILD_ROOT/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/fixinc.sh
rm $RPM_BUILD_ROOT/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/fixincl
rm $RPM_BUILD_ROOT/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/mkheaders
rm $RPM_BUILD_ROOT/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/install-tools/mkinstalldirs
rm $RPM_BUILD_ROOT/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/lto-wrapper
rm $RPM_BUILD_ROOT/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/lto1
rm $RPM_BUILD_ROOT/opt/kgen/share/info/clan.info
rm $RPM_BUILD_ROOT/opt/kgen/share/info/dir
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/bugpoint.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/dragonegg-cpp.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/dragonegg-gcc.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/dragonegg-gcov.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/dragonegg-gfortran.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/lit.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llc.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/lli.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-ar.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-as.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-bcanalyzer.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-config.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-diff.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-dis.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-extract.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-ld.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-link.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-nm.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-prof.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvm-ranlib.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvmc.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvmgcc.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/llvmgxx.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/opt.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man1/tblgen.1
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man7/fsf-funding.7
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man7/gfdl.7
rm $RPM_BUILD_ROOT/opt/kgen/share/man/man7/gpl.7


%clean
#rm -rf $RPM_BUILD_DIR/cloog
#rm -rf $RPM_BUILD_DIR/scoplib-0.2.0
#rm -rf $RPM_BUILD_DIR/llvm
#rm -rf $RPM_BUILD_DIR/gcc
#rm -rf $RPM_BUILD_DIR/dragonegg
#rm -rf $RPM_BUILD_DIR/kernelgen


%files
/opt/kgen/bin/cloog
/opt/kgen/bin/dragonegg-gfortran
/opt/kgen/bin/g95xml-refids
/opt/kgen/bin/g95xml-tree
/opt/kgen/bin/kgen
/opt/kgen/bin/kgen-ir
/opt/kgen/bin/kgen-ir-embed
/opt/kgen/bin/kgen-cpu
/opt/kgen/bin/kgen-cuda
/opt/kgen/bin/kgen-cuda-embed
/opt/kgen/bin/kgen-exec
/opt/kgen/bin/kgen-transform
/opt/kgen/bin/kgen-gfortran
/opt/kgen/bin/kgen-opencl
/opt/kgen/bin/kgen-opencl-embed
/opt/kgen/bin/llc
/opt/kgen/bin/opt
/opt/kgen/include64/kernelgen.dragonegg.mod
/opt/kgen/include64/kernelgen.h
/opt/kgen/include64/kernelgen.mod
/opt/kgen/include/kernelgen.dragonegg.mod
/opt/kgen/include/kernelgen.h
/opt/kgen/include/kernelgen.mod
/opt/kgen/lib64/dragonegg.so
/opt/kgen/lib64/libkernelgen.so
/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/collect2
/opt/kgen/libexec/gcc/x86_64-unknown-linux-gnu/4.5.4/f951
/opt/kgen/lib/libcloog-isl.so.2.0.0
/opt/kgen/lib/libkernelgen.so
/opt/kgen/lib/libisl.so
/opt/kgen/lib/libisl.so.7
/opt/kgen/lib/libisl.so.7.0.0
/opt/kgen/lib/libLLVM-3.0svn.so
/opt/kgen/lib/libscoplib.so.0.0.0
/opt/kgen/lib/LLVMPolly.so
/opt/kgen/opts/gcc.opts
/opt/kgen/opts/kgen-opencl-embed.opts
/opt/kgen/opts/nvcc.opts
/opt/kgen/transforms/split/ALGORITHM
/opt/kgen/transforms/split/cosmo-do-portable.xsl
/opt/kgen/transforms/split/de-cpp.xsl
/opt/kgen/transforms/split/device/cpu/cpu.xsl
/opt/kgen/transforms/split/device/cpu/scenes
/opt/kgen/transforms/split/device/cpu/steps
/opt/kgen/transforms/split/device/cpu/txt.output.xsl
/opt/kgen/transforms/split/device/cuda/cuda.xsl
/opt/kgen/transforms/split/device/cuda/scenes
/opt/kgen/transforms/split/device/cuda/steps
/opt/kgen/transforms/split/device/cuda/txt.output.xsl
/opt/kgen/transforms/split/device/opencl/opencl.xsl
/opt/kgen/transforms/split/device/opencl/scenes
/opt/kgen/transforms/split/device/opencl/steps
/opt/kgen/transforms/split/device/opencl/txt.output.xsl
/opt/kgen/transforms/split/device/scenes
/opt/kgen/transforms/split/device/steps
/opt/kgen/transforms/split/do-grid.xsl
/opt/kgen/transforms/split/do-group.xsl
/opt/kgen/transforms/split/do-head.xsl
/opt/kgen/transforms/split/do-links.xsl
/opt/kgen/transforms/split/do-ndims.xsl
/opt/kgen/transforms/split/do-portable.xsl
/opt/kgen/transforms/split/do-rate.xsl
/opt/kgen/transforms/split/host/cxx/cxx.xsl
/opt/kgen/transforms/split/host/cxx/scenes
/opt/kgen/transforms/split/host/cxx/steps
/opt/kgen/transforms/split/host/cxx/txt.output.xsl
/opt/kgen/transforms/split/host/fortran/fortran.xsl
/opt/kgen/transforms/split/host/fortran/scenes
/opt/kgen/transforms/split/host/fortran/steps
/opt/kgen/transforms/split/host/fortran/txt.output.xsl
/opt/kgen/transforms/split/host/scenes
/opt/kgen/transforms/split/host/site/descs.xsl
/opt/kgen/transforms/split/host/site/scenes
/opt/kgen/transforms/split/host/site/site.xsl
/opt/kgen/transforms/split/host/site/steps
/opt/kgen/transforms/split/host/site/txt.output.xsl
/opt/kgen/transforms/split/host/steps
/opt/kgen/transforms/split/scenes
/opt/kgen/transforms/split/stage.01.xsl
/opt/kgen/transforms/split/stage.02.xsl
/opt/kgen/transforms/split/stage.03.xsl
/opt/kgen/transforms/split/stage.04.xsl
/opt/kgen/transforms/split/stage.06.xsl
/opt/kgen/transforms/split/stage.07.xsl
/opt/kgen/transforms/split/stage.08.xsl
/opt/kgen/transforms/split/stage.09.xsl
/opt/kgen/transforms/split/stage.10.xsl
/opt/kgen/transforms/split/stage.11.xsl
/opt/kgen/transforms/split/stage.12.xsl
/opt/kgen/transforms/split/stage.15.xsl
/opt/kgen/transforms/split/steps

%post
echo "export PATH=\$PATH:/opt/kgen/bin" >>/etc/profile.d/kgen.sh
echo "/opt/kgen/lib" >>/etc/ld.so.conf.d/kgen.conf
echo "/opt/kgen/lib64" >>/etc/ld.so.conf.d/kgen.conf


%changelog
* Sun Jul 10 2011 Dmitry Mikushin <dmikushin@nvidia.com> 0.1
- initial release
