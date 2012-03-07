# Release name
%define release accurate

# Target operating system
%define target fedora

%if (%target == "fedora")
%define lib32 lib
%define lib64 lib64
%endif

%if (%target == "debian")
%define lib32 lib32
%define lib64 lib
%endif

# Build unoptimized version with debug info
%define debug 1

# Rebuild everything or only kernelgen
%define fullrepack 0

# The number of parallel compilation jobs
%define njobs 24

# Don't strip anything from binaries in case of debug
%if %debug
%define __os_install_post %{nil}
%endif

AutoReq: 0

Name:           kernelgen
Version:        0.2
Release:        %{release}
Summary:        Compiler with automatic generation of GPU kernels from the regular source code 
Source0:	ftp://upload.hpcforge.org/pub/kernelgen/llvm-r151057.tar.gz
Source1:	ftp://upload.hpcforge.org/pub/kernelgen/gcc-4.6.3.tar.bz2
Source2:	ftp://upload.hpcforge.org/pub/kernelgen/dragonegg-r151057.tar.gz
Source3:	ftp://upload.hpcforge.org/pub/kernelgen/kernelgen-r679.tar.gz
Source4:	ftp://upload.hpcforge.org/pub/kernelgen/polly-151057.tar.gz
Source5:	ftp://upload.hpcforge.org/pub/kernelgen/nvopencc-r12003483.tar.gz
Patch0:		llvm.varargs.patch
Patch1:		llvm.patch
Patch2:		llvm.gpu.patch
Patch3:		gcc.patch
Patch4:		gcc.opencl.patch
Patch5:		dragonegg.opencl.patch
Patch6:		dragonegg.ptx.patch
Patch7: 	nvopencc.patch
Patch8:		llvm.polly.patch
Patch8:		llvm.SCEV.patch

Group:          Applications/Engineering
License:        GPL/BSD/Freeware
URL:            https://hpcforge.org/projects/kernelgen/

%if (%target == fedora)
BuildRequires:  gcc gcc-c++ gcc-gfortran perl elfutils-libelf-devel libffi-devel gmp-devel mpfr-devel libmpc-devel flex glibc-devel git autoconf automake libtool
Requires:       elfutils-libelf libffi gmp mpfr libmpc
%else
#BuildRequires:	gcc g++ gfortran perl libelf-dev libffi-dev libgmp3-dev libmpfr-dev libmpc-dev flex libc6-dev libc6-dev-i386 gcc-multiliblib git autoconf automake libtool
#Requires:	libelf ffi libgmp3 libmpfr libmpc g++-4.6-multilib
%endif

Packager:       Dmitry Mikushin <maemarcus@gmail.com>

%description
A tool for automatic generation of GPU kernels from CPU-targeted source code. From user's point of view it acts as regular GNU-compatible compiler.


%prep
%if %fullrepack
rm -rf $RPM_BUILD_DIR/llvm
tar -xf $RPM_SOURCE_DIR/llvm-r151057.tar.gz
cd $RPM_BUILD_DIR/llvm/tools
tar -xf $RPM_SOURCE_DIR/polly-r151057.tar.gz
cd $RPM_BUILD_DIR
rm -rf $RPM_BUILD_DIR/gcc-4.6.3
tar -xjf $RPM_SOURCE_DIR/gcc-4.6.3.tar.bz2
rm -rf $RPM_BUILD_DIR/dragonegg
tar -xf $RPM_SOURCE_DIR/dragonegg-r151057.tar.gz
rm -rf $RPM_BUILD_DIR/cloog
mkdir -p $RPM_BUILD_DIR/cloog
sh $RPM_BUILD_DIR/llvm/tools/polly/utils/checkout_cloog.sh $RPM_BUILD_DIR/cloog
rm -rf $RPM_BUILD_DIR/nvopencc
tar -xf $RPM_SOURCE_DIR/nvopencc-r12003483.tar.gz
%endif
rm -rf $RPM_BUILD_DIR/kernelgen
tar -xf $RPM_SOURCE_DIR/kernelgen-r679.tar.gz


%if %fullrepack
%patch0 -p1
%patch1 -p1
%patch2 -p1
%patch3 -p1
%patch4 -p1
%patch5 -p1
%patch6 -p1
%patch7 -p1
%patch8 -p1
%patch9 -p1
%endif


%build
%if %fullrepack
%if %debug
cd $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa
make
%else
cd $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa_rel
make
%endif
cd $RPM_BUILD_DIR/cloog
./autogen.sh
./configure --prefix=$RPM_BUILD_ROOT/opt/kernelgen
make
make install
cd $RPM_BUILD_DIR/llvm
mkdir build
cp -rf include/ build/include/
cd build
%if %debug
../configure --enable-jit --enable-debug-runtime --enable-debug-symbols --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe,ptx --with-cloog=$RPM_BUILD_ROOT/opt/kernelgen --with-isl=$RPM_BUILD_ROOT/opt/kernelgen
make -j%{njobs} CXXFLAGS=-O0
%else
../configure --enable-jit --enable-optimized --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe,ptx --with-cloog=$RPM_BUILD_ROOT/opt/kernelgen --with-isl=$RPM_BUILD_ROOT/opt/kernelgen
make -j%{njobs}
%endif
cd $RPM_BUILD_DIR/gcc-4.6.3
mkdir build
cd build/
../configure --prefix=$RPM_BUILD_ROOT/opt/kernelgen --program-prefix=kernelgen- --enable-languages=fortran --with-mpfr-include=/usr/include/ --with-mpfr-lib=/usr/lib64 --with-gmp-include=/usr/include/ --with-gmp-lib=/usr/lib64 --enable-plugin
%if %debug
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu make -j%{njobs} CFLAGS="-g -O0" CXXFLAGS="-g -O0"
%else
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu  make -j%{njobs}
%endif
%endif
cd $RPM_BUILD_DIR/kernelgen
./configure
make src


%install
rm -rf $RPM_BUILD_ROOT
mkdir -p $RPM_BUILD_ROOT/opt/kernelgen/bin
mkdir -p $RPM_BUILD_ROOT/opt/kernelgen/lib
%if %debug
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa/bin/nvopencc $RPM_BUILD_ROOT/opt/kernelgen/bin/nvopencc
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa/lib/be $RPM_BUILD_ROOT/opt/kernelgen/lib/be
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa/lib/gfec $RPM_BUILD_ROOT/opt/kernelgen/lib/gfec
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa/lib/inline $RPM_BUILD_ROOT/opt/kernelgen/lib/inline
%else
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa_rel/bin/nvopencc $RPM_BUILD_ROOT/opt/kernelgen/bin/nvopencc
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa_rel/lib/be $RPM_BUILD_ROOT/opt/kernelgen/lib/be
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa_rel/lib/gfec $RPM_BUILD_ROOT/opt/kernelgen/lib/gfec
cp $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa_rel/lib/inline $RPM_BUILD_ROOT/opt/kernelgen/lib/inline
%endif
cd $RPM_BUILD_DIR/cloog
make install
cd $RPM_BUILD_DIR/llvm/build
make install
cd $RPM_BUILD_DIR/gcc-4.6.3/build
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu  make install
cd $RPM_BUILD_DIR/dragonegg
GCC=$RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcc LLVM_CONFIG=$RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config make clean
GCC=$RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcc LLVM_CONFIG=$RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config make
cp dragonegg.so $RPM_BUILD_ROOT/opt/kernelgen/%{lib64}/
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPolly.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/CodeGeneration.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/bugpoint
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-cpp
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcov
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/lli
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-ar
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-as
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-bcanalyzer
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-diff
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-dis
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-ld
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-link
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-mc
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-nm
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-objdump
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-prof
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-ranlib
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-rtdyld
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-stub
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/macho-dump
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-tblgen
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/x86_64-unknown-linux-gnu-gcc-4.6.3
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/x86_64-unknown-linux-gnu-kernelgen-gcc
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/x86_64-unknown-linux-gnu-kernelgen-gfortran
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html.tar.gz
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/AliasAnalysis.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/BitCodeFormat.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/BranchWeightMetadata.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Bugpoint.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CFEBuildInstrs.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CMake.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CodeGenerator.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CodingStandards.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/FileCheck.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/bugpoint.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/index.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/lit.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llc.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/lli.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ar.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-as.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-bcanalyzer.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-config.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-diff.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-dis.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-extract.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ld.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-link.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-nm.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-prof.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ranlib.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/manpage.css
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/opt.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-tblgen.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandLine.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CompilerDriver.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CompilerDriverTutorial.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CompilerWriterInfo.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/DebuggingJITedCode.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/DeveloperPolicy.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ExceptionHandling.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ExtendingLLVM.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/FAQ.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GCCFEBuildInstrs.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GarbageCollection.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GetElementPtr.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GettingStarted.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GettingStartedVS.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GoldPlugin.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/HowToReleaseLLVM.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/HowToSubmitABug.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/LangRef.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Lexicon.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/LinkTimeOptimization.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/MakefileGuide.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Packaging.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Passes.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ProgrammersManual.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Projects.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ReleaseNotes.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/SourceLevelDebugging.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/SystemLibrary.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/TableGenFundamentals.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/TestingGuide.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/UsingLibraries.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/WritingAnLLVMBackend.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/WritingAnLLVMPass.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/doxygen.css
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/Debugging.gif
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/libdeps.gif
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/lines.gif
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/objdeps.gif
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/venusflytrap.jpg
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/index.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/llvm.css
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl1.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl2.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl3.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl4.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl5.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl6.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl7.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl8.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl1.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl2.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl3.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl4.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl5.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl6.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl7.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl8.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/index.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/FileCheck.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/bugpoint.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/lit.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llc.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/lli.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-ar.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-as.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-bcanalyzer.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-config.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-diff.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-dis.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-extract.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-ld.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-link.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-nm.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-prof.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-ranlib.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvmc.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvmgcc.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvmgxx.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/opt.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/tblgen.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/block.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/clast.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/cloog.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/constraints.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/domain.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/input.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/int.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/backend.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/cloog.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/constraintset.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/domain.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/loop.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/matrix.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/matrix/constraintset.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/names.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/options.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/pprint.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/program.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/state.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/statement.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/stride.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/union_domain.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/version.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/aff.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/aff_type.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/arg.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/band.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/blk.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/config.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/constraint.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/ctx.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/dim.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/div.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/flow.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/hash.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/ilp.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/int.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/list.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/local_space.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/lp.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/map.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/map_type.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/mat.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/obj.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/options.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/point.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/polynomial.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/printer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/schedule.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/seq.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/set.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/set_type.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/stdint.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/stream.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/union_map.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/union_set.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/vec.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/version.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/vertices.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Analysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/BitReader.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/BitWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Core.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Disassembler.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/EnhancedDisassembly.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/ExecutionEngine.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Initialization.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/LinkTimeOptimizer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Object.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Target.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Transforms/IPO.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Transforms/Scalar.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/lto.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/APFloat.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/APInt.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/APSInt.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ArrayRef.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/BitVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DAGDeltaAlgorithm.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DeltaAlgorithm.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DenseMap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DenseMapInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DenseSet.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DepthFirstIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/EquivalenceClasses.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/FoldingSet.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/GraphTraits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableIntervalMap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableList.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableMap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableSet.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/InMemoryStruct.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IndexedMap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IntEqClasses.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IntervalMap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IntrusiveRefCntPtr.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/NullablePtr.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Optional.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/OwningPtr.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PackedVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PointerIntPair.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PointerUnion.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PostOrderIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PriorityQueue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SCCIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/STLExtras.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ScopedHashTable.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SetOperations.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SetVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallBitVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallPtrSet.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallSet.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallString.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SparseBitVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Statistic.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringExtras.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringMap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringRef.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringSet.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringSwitch.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/TinyPtrVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Trie.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Triple.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Twine.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/UniqueVector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ValueMap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/VectorExtras.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ilist.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ilist_node.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/AliasAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/AliasSetTracker.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/BlockFrequencyImpl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/BlockFrequencyInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/BranchProbabilityInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CFGPrinter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CallGraph.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CaptureTracking.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CodeMetrics.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ConstantFolding.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ConstantsScanner.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DIBuilder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DOTGraphTraitsPass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DebugInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DomPrinter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DominanceFrontier.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DominatorInternals.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Dominators.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/FindUsedTypes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/IVUsers.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/InlineCost.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/InstructionSimplify.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Interval.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/IntervalIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/IntervalPartition.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LazyValueInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LibCallAliasAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LibCallSemantics.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Lint.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Loads.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LoopDependenceAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LoopInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LoopPass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/MemoryBuiltins.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/MemoryDependenceAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PHITransAddr.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Passes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PathNumbering.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PathProfileInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PostDominators.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ProfileInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ProfileInfoLoader.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ProfileInfoTypes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionPass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionPrinter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolution.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionExpander.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionExpressions.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionNormalization.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/SparsePropagation.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Trace.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ValueTracking.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Verifier.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Argument.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/AssemblyAnnotationWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/Parser.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/PrintModulePass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/Writer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Attributes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/AutoUpgrade.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/BasicBlock.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/Archive.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/BitCodes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/BitstreamReader.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/BitstreamWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/LLVMBitCodes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/ReaderWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CallGraphSCCPass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CallingConv.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/Analysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/AsmPrinter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/BinaryObject.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/CalcSpillWeights.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/CallingConvLower.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/EdgeBundles.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/FastISel.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/FunctionLoweringInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCMetadata.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCMetadataPrinter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCStrategy.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCs.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ISDOpcodes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/IntrinsicLowering.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/JITCodeEmitter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LatencyPriorityQueue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LinkAllAsmWriterComponents.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LinkAllCodegenComponents.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveInterval.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveIntervalAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveStackAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveVariables.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachORelocation.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineBasicBlock.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineBlockFrequencyInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineBranchProbabilityInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineCodeEmitter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineCodeInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineConstantPool.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineDominators.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFrameInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFunction.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFunctionAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFunctionPass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineInstr.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineInstrBuilder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineJumpTableInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineLoopInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineLoopRanges.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineMemOperand.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineModuleInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineModuleInfoImpls.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineOperand.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachinePassRegistry.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineRegisterInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineRelocation.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineSSAUpdater.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ObjectCodeEmitter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Graph.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/HeuristicBase.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/HeuristicSolver.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Heuristics/Briggs.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Math.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Solution.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/Passes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ProcessImplicitDefs.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PseudoSourceValue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RegAllocPBQP.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RegAllocRegistry.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RegisterScavenging.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RuntimeLibcalls.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ScheduleDAG.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ScheduleHazardRecognizer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SchedulerRegistry.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ScoreboardHazardRecognizer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SelectionDAG.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SelectionDAGISel.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SelectionDAGNodes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SlotIndexes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/TargetLoweringObjectFileImpl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ValueTypes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ValueTypes.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Action.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/AutoGenerated.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/BuiltinOptions.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Common.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/CompilationGraph.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Error.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Main.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Main.inc
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Tool.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/AsmParsers.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/AsmPrinters.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/Disassemblers.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/Targets.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/config.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/llvm-config.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Constant.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Constants.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/DebugInfoProbe.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/DefaultPasses.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/DerivedTypes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/ExecutionEngine.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/GenericValue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/Interpreter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/JIT.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/JITEventListener.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/JITMemoryManager.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/MCJIT.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/RuntimeDyld.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Function.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GVMaterializer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GlobalAlias.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GlobalValue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GlobalVariable.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/InitializePasses.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/InlineAsm.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/InstrTypes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Instruction.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Instruction.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Instructions.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicInst.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Intrinsics.gen
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Intrinsics.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Intrinsics.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsARM.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsAlpha.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsCellSPU.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsPTX.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsPowerPC.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsX86.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsXCore.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/LLVMContext.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/LinkAllPasses.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/LinkAllVMCore.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Linker.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/EDInstInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmBackend.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmInfoCOFF.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmInfoDarwin.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmLayout.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAssembler.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCCodeEmitter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCCodeGenInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCContext.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCDirectives.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCDisassembler.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCDwarf.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCELFObjectWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCELFSymbolFlags.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCExpr.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCFixup.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCFixupKindInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInst.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstPrinter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstrDesc.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstrInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstrItineraries.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCLabel.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCMachOSymbolFlags.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCMachObjectWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCObjectFileInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCObjectStreamer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCObjectWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/AsmCond.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/AsmLexer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCAsmLexer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCAsmParser.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCAsmParserExtension.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCParsedAsmOperand.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCRegisterInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSection.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSectionCOFF.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSectionELF.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSectionMachO.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCStreamer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSubtargetInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSymbol.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCTargetAsmLexer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCTargetAsmParser.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCValue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCWin64EH.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MachineLocation.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/SectionKind.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/SubtargetFeature.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Metadata.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Module.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/Binary.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/COFF.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/Error.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/MachOFormat.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/MachOObject.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/ObjectFile.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/OperandTraits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Operator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Pass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassAnalysisSupport.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassManager.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassManagers.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassRegistry.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassSupport.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/AIXDataTypesFix.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/AlignOf.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Allocator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Atomic.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/BlockFrequency.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/BranchProbability.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CFG.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/COFF.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CallSite.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Capacity.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Casting.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CommandLine.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Compiler.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ConstantFolder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ConstantRange.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CrashRecoveryContext.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DOTGraphTraits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DataFlow.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DataTypes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Debug.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DebugLoc.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Disassembler.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Dwarf.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DynamicLibrary.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ELF.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Endian.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Errno.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ErrorHandling.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FEnv.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FileSystem.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FileUtilities.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Format.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FormattedStream.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/GetElementPtrTypeIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/GraphWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Host.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/IRBuilder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/IRReader.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/IncludeFile.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/InstIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/InstVisitor.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/LICENSE.TXT
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/LeakDetector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MachO.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ManagedStatic.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MathExtras.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Memory.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MemoryBuffer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MemoryObject.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Mutex.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MutexGuard.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/NoFolder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/OutputBuffer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PassManagerBuilder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PassNameParser.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Path.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PathV1.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PathV2.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PatternMatch.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PluginLoader.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PointerLikeTypeTraits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PredIteratorCache.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PrettyStackTrace.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Process.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Program.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/RWMutex.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Recycler.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/RecyclingAllocator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Regex.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Registry.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/RegistryParser.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SMLoc.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Signals.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Solaris.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SourceMgr.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/StringPool.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SwapByteOrder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SystemUtils.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TargetFolder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ThreadLocal.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Threading.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TimeValue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Timer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ToolOutputFile.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TypeBuilder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Valgrind.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ValueHandle.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Win64EH.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/circular_raw_ostream.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/raw_os_ostream.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/raw_ostream.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/system_error.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/type_traits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/SymbolTableListTraits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/Mangler.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/Target.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetCallingConv.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetCallingConv.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetData.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetELFWriterInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetFrameLowering.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetInstrInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetIntrinsicInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetJITInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetLibraryInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetLowering.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetLoweringObjectFile.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetMachine.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetOpcodes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetOptions.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetRegisterInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetRegistry.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSchedule.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSelect.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSelectionDAG.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSelectionDAGInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSubtargetInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/IPO.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/IPO/InlinerPass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Instrumentation.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Scalar.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/AddrModeMatcher.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/BasicBlockUtils.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/BasicInliner.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/BuildLibCalls.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/Cloning.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/FunctionUtils.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/Local.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/PromoteMemToReg.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/SSAUpdater.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/SSAUpdaterImpl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/UnifyFunctionExitNodes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/UnrollLoop.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/ValueMapper.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Type.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Use.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/User.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Value.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ValueSymbolTable.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Cloog.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Config/config.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Dependences.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/LinkAllPasses.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/MayAliasSet.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopDetection.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopLib.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopPass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Support/AffineSCEVIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Support/GICHelper.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Support/ScopHelper.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/TempScopInfo.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/macros.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/matrix.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/scop.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/statement.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/vector.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/BugpointPasses.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/LLVMHello.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtbegin.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtbeginS.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtbeginT.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtend.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtendS.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtfastmath.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtprec32.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtprec64.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtprec80.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgcc.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgcc_eh.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgcov.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgfortranbegin.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgfortranbegin.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtbegin.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtbeginS.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtbeginT.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtend.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtendS.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtfastmath.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtprec32.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtprec64.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtprec80.o
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/fixinc_list
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/gsyslimits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/include/README
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/include/limits.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/macro_list
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/mkheaders.conf
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgcc.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgcc_eh.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgcov.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgfortranbegin.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgfortranbegin.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ada/gcc-interface/ada-tree.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/alias.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/all-tree.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ansidecl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/auto-host.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/b-header-vars
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/basic-block.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/bitmap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/builtins.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/bversion.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-common.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-family/c-common.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-objc.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-pragma.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-pretty-print.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cfghooks.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cfgloop.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cgraph.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cif-code.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/dbxelf.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/elfos.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/glibc-stdint.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/gnu-user.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/att.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/biarch64.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/i386-protos.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/i386.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/linux64.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/unix.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/x86-64.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/linux-android.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/linux.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/vxworks-dummy.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/configargs.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/coretypes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cp/cp-tree.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cppdefault.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cpplib.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/debug.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/defaults.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/diagnostic-core.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/diagnostic.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/diagnostic.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/double-int.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/emit-rtl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/except.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/filenames.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/fixed-value.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/flag-types.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/flags.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/function.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gcc-plugin.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/genrtl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ggc.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gimple.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gimple.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gsstruct.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gtype-desc.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/hard-reg-set.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/hashtab.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/highlev-plugin-common.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/hwint.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/incpath.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/input.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-constants.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-flags.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-modes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-notes.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/intl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-prop.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-ref-inline.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-ref.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-reference.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-utils.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/java/java-tree.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/langhooks.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/libiberty.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/line-map.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/machmode.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/md5.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/mode-classes.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/objc/objc-tree.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/obstack.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/omp-builtins.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/options.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/opts.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/output.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/params.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/params.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin-api.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin-version.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/pointer-set.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/predict.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/predict.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/prefix.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/pretty-print.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/real.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/reg-notes.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/rtl.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/rtl.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/safe-ctype.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/sbitmap.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/splay-tree.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/statistics.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/symtab.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/sync-builtins.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/system.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/target.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/target.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/timevar.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/timevar.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tm-preds.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tm.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tm_p.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/toplev.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-check.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-dump.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-flow-inline.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-flow.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-inline.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-iterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-pass.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-ssa-alias.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-ssa-operands.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-ssa-sccvn.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/treestruct.def
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/vec.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/vecir.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/vecprim.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/version.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libCompilerDriver.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libEnhancedDisassembly.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libEnhancedDisassembly.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMAnalysis.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMArchive.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMAsmParser.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMAsmPrinter.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMBitReader.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMBitWriter.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCBackend.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCBackendInfo.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCodeGen.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCore.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMExecutionEngine.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMInstCombine.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMInstrumentation.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMInterpreter.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMJIT.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMLinker.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMC.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMCDisassembler.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMCJIT.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMCParser.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMObject.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMRuntimeDyld.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMScalarOpts.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMSelectionDAG.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMSupport.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMTarget.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMTransformUtils.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86AsmParser.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86AsmPrinter.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86CodeGen.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Desc.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Disassembler.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Info.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Utils.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMipa.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMipo.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLTO.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLTO.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libcloog-isl.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libcloog-isl.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib64}/libiberty.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libisl.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libisl.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libisl.so.7.0.0-gdb.py
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollyanalysis.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollyexchange.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollyjson.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollysupport.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libprofile_rt.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libprofile_rt.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libscoplib.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libscoplib.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libscoplib.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/pkgconfig/cloog-isl.pc
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/pkgconfig/isl.pc
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgcc_s.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgcc_s.so.1
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.so.3
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.so.3.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.spec
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.so.1
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.so.1.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.spec
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp_nonshared.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp_nonshared.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgcc_s.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgcc_s.so.1
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.so.3
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.so.3.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.spec
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.so.1
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.so.1.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.spec
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.so
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.so.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.so.0.0.0
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp_nonshared.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp_nonshared.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/fixinc.sh
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/fixincl
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/mkheaders
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/mkinstalldirs
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.la
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/lto-wrapper
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/lto1
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/X11/Xw32defs.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/openssl/bn.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPTXCodeGen.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPTXDesc.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPTXInfo.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-cov
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-dwarfdump
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-size
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Atomics.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-build.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-cov.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/tblgen.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/HowToAddABuilder.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/LLVMBuild.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/SegmentedStacks.html
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl5-cfg.png
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-build.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-cov.ps
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/id.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/multi.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/polynomial_type.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/isl/space.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Transforms/PassManagerBuilder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Transforms/Vectorize.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Hashing.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/VariadicFunction.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/edit_distance.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LoopIterator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/DFAPacketizer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LexicalScopes.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineInstrBundle.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ResourcePriorityQueue.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/DebugInfo/DIContext.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsHexagon.td
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAtom.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstrAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCModule.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCWinCOFFObjectWriter.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/Archive.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/ELF.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/MachO.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CodeGen.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DataExtractor.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DataStream.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/GCOV.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/JSONParser.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/LockFileManager.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/StreamableMemoryObject.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TargetRegistry.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TargetSelect.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/TableGen/Error.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/TableGen/Main.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/TableGen/Record.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/TableGen/TableGenAction.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/TableGen/TableGenBackend.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/IPO/PassManagerBuilder.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/CmpInstAnalysis.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/ModuleUtils.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/SimplifyIndVar.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Vectorize.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/RegisterPasses.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScheduleOptimizer.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Support/SCEVValidator.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCBackendCodeGen.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMDebugInfo.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPTXAsmPrinter.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMTableGen.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMVectorize.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libisl.so.9.0.0-gdb.py
cd $RPM_BUILD_DIR/kernelgen
ROOT=$RPM_BUILD_ROOT LIB32=%{lib32} LIB64=%{lib64} make install

%clean
#rm -rf $RPM_BUILD_DIR/cloog
#rm -rf $RPM_BUILD_DIR/scoplib-0.2.0
#rm -rf $RPM_BUILD_DIR/llvm
#rm -rf $RPM_BUILD_DIR/gcc
#rm -rf $RPM_BUILD_DIR/dragonegg
#rm -rf $RPM_BUILD_DIR/kernelgen


%files
/opt/kernelgen/bin/cloog
/opt/kernelgen/bin/kernelgen
/opt/kernelgen/bin/kernelgen-gcc
/opt/kernelgen/bin/kernelgen-gfortran
/opt/kernelgen/bin/llc
/opt/kernelgen/bin/opt
/opt/kernelgen/bin/llvm-extract
/opt/kernelgen/bin/nvopencc
/opt/kernelgen/lib/be
/opt/kernelgen/lib/gfec
/opt/kernelgen/lib/inline
/opt/kernelgen/include/kernelgen_interop.h
/opt/kernelgen/include/kernelgen_memory.h
/opt/kernelgen/include/kernelgen_runtime.h
/opt/kernelgen/%{lib64}/dragonegg.so
/opt/kernelgen/%{lib64}/libkernelgen.a
/opt/kernelgen/%{lib64}/libasfermi.so
/opt/kernelgen/%{lib64}/libdyloader.so
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/cc1
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/collect2
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/f951
/opt/kernelgen/lib/LLVMPolly.so
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.so
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.so.0
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.so.0.0.0
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib.f90
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib.kernelgen.mod
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib_kinds.kernelgen.mod
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/README
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/limits.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/linux/a.out.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/syslimits.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/abmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/ammintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/avxintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/bmiintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/bmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/cpuid.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/cross-stdarg.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/emmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/float.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/fma4intrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/ia32intrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/immintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/iso646.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/lwpintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/mf-runtime.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/mm3dnow.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/mm_malloc.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/mmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/nmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/omp.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/pmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/popcntintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/quadmath.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/quadmath_weak.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/smmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/ssp/ssp.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/ssp/stdio.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/ssp/string.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/ssp/unistd.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/stdarg.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/stdbool.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/stddef.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/stdfix.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/stdint-gcc.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/stdint.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/tbmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/tmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/unwind.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/varargs.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/wmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/x86intrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/xmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include/xopintrin.h
/opt/kernelgen/lib/libLLVM-3.1svn.so
/opt/kernelgen/lib/libcloog-isl.so
/opt/kernelgen/lib/libcloog-isl.so.3
/opt/kernelgen/lib/libcloog-isl.so.3.0.0
/opt/kernelgen/lib/libisl.so
/opt/kernelgen/lib/libisl.so.9
/opt/kernelgen/lib/libisl.so.9.0.0

%post
echo "export PATH=\$PATH:/opt/kernelgen/bin" >>/etc/profile.d/kernelgen.sh
echo "/opt/kernelgen/lib" >>/etc/ld.so.conf.d/kernelgen.conf
echo "/opt/kernelgen/lib64" >>/etc/ld.so.conf.d/kernelgen.conf


%changelog
* Tue Sep 13 2011 Dmitry Mikushin <maemarcus@gmail.com> 0.2
- started preparing 0.2 "accurate" release
* Sun Jul 10 2011 Dmitry Mikushin <dmikushin@nvidia.com> 0.1
- initial release

