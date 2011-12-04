# Release name
%define release accurate

# Target operating system
%define target debian

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

AutoReq: 0

Name:           kernelgen
Version:        0.2
Release:        %{release}
Summary:        Compiler with automatic generation of GPU kernels from the regular source code 
Source0:	ftp://upload.hpcforge.org/pub/kernelgen/llvm-r136600.tar.gz
Source1:	ftp://upload.hpcforge.org/pub/kernelgen/gcc-4.6-r178876.tar.gz
Source2:	ftp://upload.hpcforge.org/pub/kernelgen/dragonegg-r136347.tar.gz
Source3:	ftp://upload.hpcforge.org/pub/kernelgen/kernelgen-r544.tar.gz
Source4:	ftp://upload.hpcforge.org/pub/kernelgen/polly-r137304.tar.gz
Source5:	ftp://upload.hpcforge.org/pub/kernelgen/cloog-225c2ed62fe37a4db22bf4b95c3731dab1a50dde.tar.gz
Source6:	ftp://upload.hpcforge.org/pub/kernelgen/scoplib-0.2.0.tar.gz
Source7:	ftp://upload.hpcforge.org/pub/kernelgen/nvopencc-r11207187.tar.gz
Patch0:		llvm.varargs.patch
Patch1:		llvm.patch
Patch2:		llvm.gpu.patch
Patch3:		gcc.patch
Patch4:		gcc.opencl.patch
Patch5:		dragonegg.opencl.patch
Patch6:		dragonegg.ptx.patch

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
A tool for automatic generation of GPU kernels from Fortran source code. From user's point of view it acts as regular GNU-compatible compiler.


%prep
%if %fullrepack
rm -rf $RPM_BUILD_DIR/llvm
tar -xf $RPM_SOURCE_DIR/llvm-r136600.tar.gz
cd $RPM_BUILD_DIR/llvm/tools
tar -xf $RPM_SOURCE_DIR/polly-r137304.tar.gz
cd $RPM_BUILD_DIR
rm -rf $RPM_BUILD_DIR/gcc-4.6
tar -xf $RPM_SOURCE_DIR/gcc-4.6-r178876.tar.gz
rm -rf $RPM_BUILD_DIR/dragonegg
tar -xf $RPM_SOURCE_DIR/dragonegg-r136347.tar.gz
rm -rf $RPM_BUILD_DIR/cloog
tar -xf $RPM_SOURCE_DIR/cloog-225c2ed62fe37a4db22bf4b95c3731dab1a50dde.tar.gz
rm -rf $RPM_BUILD_DIR/scoplib-0.2.0
tar -xf $RPM_SOURCE_DIR/scoplib-0.2.0.tar.gz
rm -rf $RPM_BUILD_DIR/nvopencc
tar -xf $RPM_SOURCE_DIR/nvopencc-r11207187.tar.gz
%endif
rm -rf $RPM_BUILD_DIR/kernelgen
tar -xf $RPM_SOURCE_DIR/kernelgen-r544.tar.gz


%if %fullrepack
%patch0 -p1
%patch1 -p1
%patch2 -p1
%patch3 -p1
%patch4 -p1
%patch5 -p1
%patch6 -p1
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
./get_submodules.sh
./autogen.sh
./configure --prefix=$RPM_BUILD_ROOT/opt/kernelgen
make
ln -s $RPM_BUILD_DIR/cloog/.libs $RPM_BUILD_DIR/cloog/lib
ln -s $RPM_BUILD_DIR/cloog/isl/.libs $RPM_BUILD_DIR/cloog/isl/lib
cd $RPM_BUILD_DIR/scoplib-0.2.0
./configure --enable-mp-version --prefix=$RPM_BUILD_ROOT/opt/kernelgen
make
ln -s $RPM_BUILD_DIR/scoplib-0.2.0/source/.libs $RPM_BUILD_DIR/scoplib-0.2.0/lib
cd $RPM_BUILD_DIR/llvm
mkdir build
cp -rf include/ build/include/
cd build
%if %debug
../configure --enable-jit --enable-debug-runtime --enable-debug-symbols --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe,ptx --with-cloog=$RPM_BUILD_DIR/cloog --with-isl=$RPM_BUILD_DIR/cloog/isl --with-scoplib=$RPM_BUILD_DIR/scoplib-0.2.0
make -j%{njobs} CXXFLAGS=-O0
%else
../configure --enable-jit --enable-optimized --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe,ptx --with-cloog=$RPM_BUILD_DIR/cloog --with-isl=$RPM_BUILD_DIR/cloog/isl --with-scoplib=$RPM_BUILD_DIR/scoplib-0.2.0
make -j%{njobs}
%endif
cd $RPM_BUILD_DIR/gcc-4.6
mkdir build
cd build/
../configure --prefix=$RPM_BUILD_ROOT/opt/kernelgen --program-prefix=kernelgen- --enable-languages=fortran --with-mpfr-include=/usr/include/ --with-mpfr-lib=/usr/lib64 --with-gmp-include=/usr/include/ --with-gmp-lib=/usr/lib64 --enable-plugin
%if %debug
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu make -j%{njobs} CFLAGS="-g -O0" CXXFLAGS="-g -O0"
%else
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu  make -j%{njobs}
%endif
%endif
cd $RPM_BUILD_DIR/kernelgen/branches/accurate
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
cd $RPM_BUILD_DIR/scoplib-0.2.0
make install
cd $RPM_BUILD_DIR/llvm/build
make install
cd $RPM_BUILD_DIR/gcc-4.6/build
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu  make install
cd $RPM_BUILD_DIR/dragonegg
GCC=$RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcc LLVM_CONFIG=$RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config make clean
GCC=$RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcc LLVM_CONFIG=$RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config make
cp dragonegg.so $RPM_BUILD_ROOT/opt/kernelgen/%{lib64}/
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPolly.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/CodeGeneration.h
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/bugpoint
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-cpp
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcov
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/lli
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-ar
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-as
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-bcanalyzer
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-diff
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-dis
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-ld
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-link
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-mc
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-nm
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-objdump
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-prof
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-ranlib
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-rtdyld
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-stub
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/llvmc
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/macho-dump
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/tblgen
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/x86_64-unknown-linux-gnu-gcc-4.6.2
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/x86_64-unknown-linux-gnu-kernelgen-gcc
rm $RPM_BUILD_ROOT/opt/kernelgen/bin/x86_64-unknown-linux-gnu-kernelgen-gfortran
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html.tar.gz
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/AliasAnalysis.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/BitCodeFormat.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/BranchWeightMetadata.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Bugpoint.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CFEBuildInstrs.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CMake.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CodeGenerator.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CodingStandards.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/FileCheck.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/bugpoint.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/index.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/lit.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llc.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/lli.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ar.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-as.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-bcanalyzer.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-config.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-diff.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-dis.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-extract.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ld.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-link.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-nm.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-prof.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ranlib.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvmc.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvmgcc.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/llvmgxx.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/manpage.css
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/opt.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandGuide/tblgen.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CommandLine.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CompilerDriver.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CompilerDriverTutorial.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/CompilerWriterInfo.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/DebuggingJITedCode.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/DeveloperPolicy.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ExceptionHandling.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ExtendingLLVM.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/FAQ.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GCCFEBuildInstrs.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GarbageCollection.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GetElementPtr.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GettingStarted.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GettingStartedVS.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/GoldPlugin.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/HowToReleaseLLVM.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/HowToSubmitABug.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/LangRef.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Lexicon.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/LinkTimeOptimization.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/MakefileGuide.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Packaging.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Passes.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ProgrammersManual.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/Projects.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/ReleaseNotes.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/SourceLevelDebugging.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/SystemLibrary.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/TableGenFundamentals.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/TestingGuide.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/UsingLibraries.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/WritingAnLLVMBackend.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/WritingAnLLVMPass.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/doxygen.css
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/Debugging.gif
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/libdeps.gif
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/lines.gif
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/objdeps.gif
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/img/venusflytrap.jpg
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/index.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/llvm.css
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl1.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl2.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl3.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl4.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl5.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl6.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl7.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/LangImpl8.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl1.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl2.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl3.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl4.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl5.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl6.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl7.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl8.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/html/tutorial/index.html
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/FileCheck.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/bugpoint.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/lit.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llc.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/lli.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-ar.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-as.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-bcanalyzer.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-config.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-diff.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-dis.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-extract.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-ld.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-link.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-nm.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-prof.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvm-ranlib.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvmc.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvmgcc.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/llvmgxx.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/opt.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/docs/llvm/ps/tblgen.ps
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/block.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/clast.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/cloog.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/constraints.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/domain.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/input.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/int.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/backend.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/cloog.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/constraintset.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/isl/domain.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/loop.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/matrix.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/matrix/constraintset.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/names.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/options.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/pprint.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/program.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/state.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/statement.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/stride.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/union_domain.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/cloog/version.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/aff.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/aff_type.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/arg.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/band.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/blk.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/config.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/constraint.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/ctx.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/dim.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/div.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/flow.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/hash.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/ilp.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/int.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/list.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/local_space.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/lp.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/map.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/map_type.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/mat.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/obj.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/options.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/point.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/polynomial.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/printer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/schedule.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/seq.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/set.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/set_type.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/stdint.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/stream.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/union_map.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/union_set.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/vec.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/version.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/isl/vertices.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Analysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/BitReader.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/BitWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Core.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Disassembler.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/EnhancedDisassembly.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/ExecutionEngine.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Initialization.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/LinkTimeOptimizer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Object.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Target.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Transforms/IPO.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/Transforms/Scalar.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm-c/lto.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/APFloat.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/APInt.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/APSInt.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ArrayRef.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/BitVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DAGDeltaAlgorithm.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DeltaAlgorithm.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DenseMap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DenseMapInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DenseSet.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/DepthFirstIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/EquivalenceClasses.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/FoldingSet.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/GraphTraits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableIntervalMap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableList.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableMap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ImmutableSet.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/InMemoryStruct.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IndexedMap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IntEqClasses.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IntervalMap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/IntrusiveRefCntPtr.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/NullablePtr.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Optional.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/OwningPtr.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PackedVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PointerIntPair.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PointerUnion.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PostOrderIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/PriorityQueue.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SCCIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/STLExtras.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ScopedHashTable.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SetOperations.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SetVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallBitVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallPtrSet.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallSet.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallString.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SmallVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/SparseBitVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Statistic.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringExtras.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringMap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringRef.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringSet.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/StringSwitch.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/TinyPtrVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Trie.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Triple.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/Twine.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/UniqueVector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ValueMap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/VectorExtras.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ilist.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ADT/ilist_node.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/AliasAnalysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/AliasSetTracker.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/BlockFrequencyImpl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/BlockFrequencyInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/BranchProbabilityInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CFGPrinter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CallGraph.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CaptureTracking.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/CodeMetrics.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ConstantFolding.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ConstantsScanner.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DIBuilder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DOTGraphTraitsPass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DebugInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DomPrinter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DominanceFrontier.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/DominatorInternals.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Dominators.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/FindUsedTypes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/IVUsers.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/InlineCost.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/InstructionSimplify.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Interval.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/IntervalIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/IntervalPartition.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LazyValueInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LibCallAliasAnalysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LibCallSemantics.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Lint.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Loads.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LoopDependenceAnalysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LoopInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/LoopPass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/MemoryBuiltins.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/MemoryDependenceAnalysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PHITransAddr.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Passes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PathNumbering.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PathProfileInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/PostDominators.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ProfileInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ProfileInfoLoader.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ProfileInfoTypes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionPass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/RegionPrinter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolution.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionExpander.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionExpressions.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionNormalization.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/SparsePropagation.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Trace.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/ValueTracking.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Analysis/Verifier.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Argument.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/AssemblyAnnotationWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/Parser.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/PrintModulePass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Assembly/Writer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Attributes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/AutoUpgrade.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/BasicBlock.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/Archive.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/BitCodes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/BitstreamReader.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/BitstreamWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/LLVMBitCodes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Bitcode/ReaderWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CallGraphSCCPass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CallingConv.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/Analysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/AsmPrinter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/BinaryObject.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/CalcSpillWeights.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/CallingConvLower.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/EdgeBundles.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/FastISel.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/FunctionLoweringInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCMetadata.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCMetadataPrinter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCStrategy.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/GCs.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ISDOpcodes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/IntrinsicLowering.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/JITCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LatencyPriorityQueue.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LinkAllAsmWriterComponents.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LinkAllCodegenComponents.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveInterval.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveIntervalAnalysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveStackAnalysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/LiveVariables.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachORelocation.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineBasicBlock.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineBlockFrequencyInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineBranchProbabilityInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineCodeInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineConstantPool.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineDominators.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFrameInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFunction.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFunctionAnalysis.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineFunctionPass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineInstr.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineInstrBuilder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineJumpTableInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineLoopInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineLoopRanges.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineMemOperand.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineModuleInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineModuleInfoImpls.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineOperand.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachinePassRegistry.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineRegisterInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineRelocation.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/MachineSSAUpdater.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ObjectCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Graph.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/HeuristicBase.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/HeuristicSolver.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Heuristics/Briggs.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Math.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PBQP/Solution.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/Passes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ProcessImplicitDefs.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/PseudoSourceValue.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RegAllocPBQP.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RegAllocRegistry.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RegisterScavenging.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/RuntimeLibcalls.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ScheduleDAG.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ScheduleHazardRecognizer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SchedulerRegistry.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ScoreboardHazardRecognizer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SelectionDAG.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SelectionDAGISel.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SelectionDAGNodes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/SlotIndexes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/TargetLoweringObjectFileImpl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ValueTypes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CodeGen/ValueTypes.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Action.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/AutoGenerated.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/BuiltinOptions.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Common.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/CompilationGraph.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Error.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Main.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Main.inc
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/CompilerDriver/Tool.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/AsmParsers.def
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/AsmPrinters.def
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/Disassemblers.def
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/Targets.def
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/config.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Config/llvm-config.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Constant.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Constants.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/DebugInfoProbe.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/DefaultPasses.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/DerivedTypes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/ExecutionEngine.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/GenericValue.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/Interpreter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/JIT.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/JITEventListener.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/JITMemoryManager.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/MCJIT.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ExecutionEngine/RuntimeDyld.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Function.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GVMaterializer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GlobalAlias.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GlobalValue.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/GlobalVariable.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/InitializePasses.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/InlineAsm.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/InstrTypes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Instruction.def
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Instruction.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Instructions.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicInst.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Intrinsics.gen
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Intrinsics.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Intrinsics.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsARM.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsAlpha.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsCellSPU.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsPTX.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsPowerPC.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsX86.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/IntrinsicsXCore.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/LLVMContext.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/LinkAllPasses.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/LinkAllVMCore.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Linker.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/EDInstInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmBackend.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmInfoCOFF.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmInfoDarwin.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAsmLayout.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCAssembler.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCCodeEmitter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCCodeGenInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCContext.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCDirectives.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCDisassembler.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCDwarf.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCELFObjectWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCELFSymbolFlags.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCExpr.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCFixup.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCFixupKindInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInst.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstPrinter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstrDesc.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstrInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCInstrItineraries.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCLabel.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCMachOSymbolFlags.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCMachObjectWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCObjectFileInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCObjectStreamer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCObjectWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/AsmCond.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/AsmLexer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCAsmLexer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCAsmParser.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCAsmParserExtension.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCParser/MCParsedAsmOperand.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCRegisterInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSection.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSectionCOFF.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSectionELF.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSectionMachO.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCStreamer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSubtargetInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCSymbol.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCTargetAsmLexer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCTargetAsmParser.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCValue.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MCWin64EH.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/MachineLocation.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/SectionKind.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/MC/SubtargetFeature.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Metadata.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Module.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/Binary.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/COFF.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/Error.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/MachOFormat.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/MachOObject.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Object/ObjectFile.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/OperandTraits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Operator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Pass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassAnalysisSupport.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassManager.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassManagers.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassRegistry.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/PassSupport.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/AIXDataTypesFix.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/AlignOf.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Allocator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Atomic.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/BlockFrequency.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/BranchProbability.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CFG.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/COFF.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CallSite.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Capacity.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Casting.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CommandLine.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Compiler.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ConstantFolder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ConstantRange.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/CrashRecoveryContext.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DOTGraphTraits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DataFlow.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DataTypes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Debug.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DebugLoc.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Disassembler.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Dwarf.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/DynamicLibrary.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ELF.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Endian.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Errno.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ErrorHandling.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FEnv.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FileSystem.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FileUtilities.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Format.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/FormattedStream.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/GetElementPtrTypeIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/GraphWriter.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Host.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/IRBuilder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/IRReader.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/IncludeFile.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/InstIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/InstVisitor.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/LICENSE.TXT
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/LeakDetector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MachO.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ManagedStatic.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MathExtras.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Memory.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MemoryBuffer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MemoryObject.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Mutex.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/MutexGuard.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/NoFolder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/OutputBuffer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PassManagerBuilder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PassNameParser.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Path.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PathV1.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PathV2.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PatternMatch.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PluginLoader.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PointerLikeTypeTraits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PredIteratorCache.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/PrettyStackTrace.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Process.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Program.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/RWMutex.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Recycler.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/RecyclingAllocator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Regex.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Registry.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/RegistryParser.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SMLoc.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Signals.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Solaris.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SourceMgr.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/StringPool.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SwapByteOrder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/SystemUtils.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TargetFolder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ThreadLocal.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Threading.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TimeValue.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Timer.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ToolOutputFile.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/TypeBuilder.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Valgrind.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/ValueHandle.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/Win64EH.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/circular_raw_ostream.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/raw_os_ostream.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/raw_ostream.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/system_error.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Support/type_traits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/SymbolTableListTraits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/Mangler.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/Target.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetCallingConv.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetCallingConv.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetData.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetELFWriterInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetFrameLowering.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetInstrInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetIntrinsicInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetJITInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetLibraryInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetLowering.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetLoweringObjectFile.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetMachine.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetOpcodes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetOptions.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetRegisterInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetRegistry.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSchedule.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSelect.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSelectionDAG.td
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSelectionDAGInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Target/TargetSubtargetInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/IPO.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/IPO/InlinerPass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Instrumentation.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Scalar.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/AddrModeMatcher.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/BasicBlockUtils.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/BasicInliner.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/BuildLibCalls.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/Cloning.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/FunctionUtils.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/Local.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/PromoteMemToReg.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/SSAUpdater.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/SSAUpdaterImpl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/UnifyFunctionExitNodes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/UnrollLoop.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Transforms/Utils/ValueMapper.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Type.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Use.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/User.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/Value.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/llvm/ValueSymbolTable.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Cloog.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Config/config.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Dependences.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/LinkAllPasses.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/MayAliasSet.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopDetection.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopLib.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/ScopPass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Support/AffineSCEVIterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Support/GICHelper.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/Support/ScopHelper.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/polly/TempScopInfo.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/macros.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/matrix.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/scop.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/statement.h
rm $RPM_BUILD_ROOT/opt/kernelgen/include/scoplib/vector.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/BugpointPasses.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/LLVMHello.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtbegin.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtbeginS.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtbeginT.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtend.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtendS.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtfastmath.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtprec32.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtprec64.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/crtprec80.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/libgcc.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/libgcc_eh.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/libgcov.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/libgfortranbegin.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/32/libgfortranbegin.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtbegin.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtbeginS.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtbeginT.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtend.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtendS.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtfastmath.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtprec32.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtprec64.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/crtprec80.o
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/fixinc_list
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/gsyslimits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/include/README
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/include/limits.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/macro_list
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/mkheaders.conf
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/libgcc.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/libgcc_eh.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/libgcov.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/libgfortranbegin.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/libgfortranbegin.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ada/gcc-interface/ada-tree.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/alias.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/all-tree.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ansidecl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/auto-host.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/b-header-vars
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/basic-block.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/bitmap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/builtins.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/bversion.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/c-common.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/c-family/c-common.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/c-objc.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/c-pragma.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/c-pretty-print.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/cfghooks.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/cfgloop.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/cgraph.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/cif-code.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/dbxelf.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/elfos.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/glibc-stdint.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/gnu-user.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/i386/att.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/i386/biarch64.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/i386/i386-protos.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/i386/i386.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/i386/linux64.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/i386/unix.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/i386/x86-64.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/linux-android.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/linux.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/config/vxworks-dummy.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/configargs.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/coretypes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/cp/cp-tree.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/cppdefault.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/cpplib.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/debug.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/defaults.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/diagnostic-core.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/diagnostic.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/diagnostic.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/double-int.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/emit-rtl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/except.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/filenames.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/fixed-value.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/flag-types.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/flags.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/function.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/gcc-plugin.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/genrtl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ggc.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/gimple.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/gimple.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/gsstruct.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/gtype-desc.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/hard-reg-set.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/hashtab.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/highlev-plugin-common.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/hwint.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/incpath.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/input.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/insn-constants.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/insn-flags.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/insn-modes.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/insn-notes.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/intl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ipa-prop.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ipa-ref-inline.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ipa-ref.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ipa-reference.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/ipa-utils.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/java/java-tree.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/langhooks.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/libiberty.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/line-map.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/machmode.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/md5.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/mode-classes.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/objc/objc-tree.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/obstack.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/omp-builtins.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/options.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/opts.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/output.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/params.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/params.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/plugin-api.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/plugin-version.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/plugin.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/plugin.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/pointer-set.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/predict.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/predict.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/prefix.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/pretty-print.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/real.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/reg-notes.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/rtl.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/rtl.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/safe-ctype.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/sbitmap.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/splay-tree.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/statistics.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/symtab.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/sync-builtins.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/system.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/target.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/target.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/timevar.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/timevar.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tm-preds.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tm.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tm_p.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/toplev.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-check.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-dump.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-flow-inline.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-flow.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-inline.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-iterator.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-pass.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-ssa-alias.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-ssa-operands.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree-ssa-sccvn.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/tree.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/treestruct.def
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/vec.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/vecir.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/vecprim.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/plugin/include/version.h
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libCompilerDriver.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libEnhancedDisassembly.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libEnhancedDisassembly.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMAnalysis.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMArchive.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMAsmParser.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMAsmPrinter.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMBitReader.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMBitWriter.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCBackend.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCBackendInfo.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCodeGen.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMCore.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMExecutionEngine.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMInstCombine.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMInstrumentation.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMInterpreter.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMJIT.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMLinker.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMC.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMCDisassembler.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMCJIT.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMMCParser.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMObject.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMRuntimeDyld.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMScalarOpts.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMSelectionDAG.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMSupport.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMTarget.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMTransformUtils.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86AsmParser.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86AsmPrinter.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86CodeGen.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Desc.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Disassembler.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Info.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMX86Utils.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMipa.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMipo.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLTO.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libLTO.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libcloog-isl.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libcloog-isl.la
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib64}/libiberty.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libisl.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libisl.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libisl.so.7.0.0-gdb.py
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollyanalysis.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollyexchange.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollyjson.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libpollysupport.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libprofile_rt.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libprofile_rt.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libscoplib.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libscoplib.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libscoplib.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/pkgconfig/cloog-isl.pc
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/pkgconfig/isl.pc
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgcc_s.so
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgcc_s.so.1
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.a
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.la
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.so
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.so.3
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.so.3.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgfortran.spec
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.a
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.la
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.so
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.so.1
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.so.1.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libgomp.spec
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.a
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.la
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.so
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflap.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.a
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.la
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.so
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libmudflapth.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.a
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.la
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.so
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libquadmath.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.a
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.la
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.so
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp_nonshared.a
rm $RPM_BUILD_ROOT/opt/kernelgen/%{lib32}/libssp_nonshared.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgcc_s.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgcc_s.so.1
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.so.3
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.so.3.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgfortran.spec
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.so.1
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.so.1.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libgomp.spec
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflap.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libmudflapth.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libquadmath.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.la
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.so.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp.so.0.0.0
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp_nonshared.a
rm $RPM_BUILD_ROOT/opt/kernelgen/lib64/libssp_nonshared.la
rm $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/fixinc.sh
rm $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/fixincl
rm $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/mkheaders
rm $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/install-tools/mkinstalldirs
rm $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/liblto_plugin.la
rm $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/lto-wrapper
rm $RPM_BUILD_ROOT/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/lto1
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include-fixed/X11/Xw32defs.h
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPTXCodeGen.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPTXDesc.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPTXInfo.a
cd $RPM_BUILD_DIR/kernelgen/branches/accurate
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
/opt/kernelgen/include/kernelgen_runtime.h
/opt/kernelgen/%{lib64}/dragonegg.so
/opt/kernelgen/%{lib64}/libkernelgen.a
/opt/kernelgen/%{lib32}/libkernelgen.a
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/cc1
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/collect2
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/f951
/opt/kernelgen/lib/libcloog-isl.so.2.0.0
/opt/kernelgen/lib/libcloog-isl.so
/opt/kernelgen/lib/libcloog-isl.so.2
/opt/kernelgen/lib/libisl.so
/opt/kernelgen/lib/libisl.so.7
/opt/kernelgen/lib/libisl.so.7.0.0
/opt/kernelgen/lib/libLLVM-3.0svn.so
/opt/kernelgen/lib/libscoplib.so.0
/opt/kernelgen/lib/libscoplib.so.0.0.0
/opt/kernelgen/lib/LLVMPolly.so
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/liblto_plugin.so
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/liblto_plugin.so.0
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/liblto_plugin.so.0.0.0
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/finclude/omp_lib.f90
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/finclude/omp_lib.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/finclude/omp_lib.kernelgen.mod
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/finclude/omp_lib_kinds.kernelgen.mod
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include-fixed/README
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include-fixed/limits.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include-fixed/linux/a.out.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include-fixed/syslimits.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/abmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/ammintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/avxintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/bmiintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/bmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/cpuid.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/cross-stdarg.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/emmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/float.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/fma4intrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/ia32intrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/immintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/iso646.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/lwpintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/mf-runtime.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/mm3dnow.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/mm_malloc.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/mmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/nmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/omp.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/pmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/popcntintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/quadmath.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/quadmath_weak.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/smmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/ssp/ssp.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/ssp/stdio.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/ssp/string.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/ssp/unistd.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/stdarg.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/stdbool.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/stddef.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/stdfix.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/stdint-gcc.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/stdint.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/tbmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/tmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/unwind.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/varargs.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/wmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/x86intrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/xmmintrin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.2/include/xopintrin.h

%post
echo "export PATH=\$PATH:/opt/kernelgen/bin" >>/etc/profile.d/kernelgen.sh
echo "/opt/kernelgen/lib" >>/etc/ld.so.conf.d/kernelgen.conf
echo "/opt/kernelgen/lib64" >>/etc/ld.so.conf.d/kernelgen.conf


%changelog
* Tue Sep 13 2011 Dmitry Mikushin <maemarcus@gmail.com> 0.2
- started preparing 0.2 "accurate" release
* Sun Jul 10 2011 Dmitry Mikushin <dmikushin@nvidia.com> 0.1
- initial release

