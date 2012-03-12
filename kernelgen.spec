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
%define fullrepack 1

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
Source3:	ftp://upload.hpcforge.org/pub/kernelgen/kernelgen-r698.tar.gz
Source4:	ftp://upload.hpcforge.org/pub/kernelgen/polly-151057.tar.gz
Source5:	ftp://upload.hpcforge.org/pub/kernelgen/nvopencc-r12003483.tar.gz
Patch0:		llvm.varargs.patch
Patch1:		llvm.patch
Patch2:		llvm.gpu.patch
Patch3:		dragonegg.ptx.patch
Patch4:		dragonegg.patch
Patch5: 	nvopencc.patch
Patch6:		llvm.polly.patch
Patch7:		llvm.scev.patch
Patch8:		gcc.patch

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


#
# Remove old files, unpack fresh content from source archives.
#
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


#
# Apply all source code patches prior to configuring.
#
%if %fullrepack
%patch0 -p1
%patch1 -p1
%patch2 -p1
%patch3 -p1
%patch4 -p1
%patch5 -p1
%patch6 -p1
%patch7 -p1
%endif


%build
%if %fullrepack
#
# Configure and build CLooG.
# Build CLooG now, as it is going to be used during LLVM's configure step.
#
cd $RPM_BUILD_DIR/cloog
./autogen.sh
./configure --prefix=$RPM_BUILD_ROOT/opt/kernelgen
make
make install
#
# Configure LLVM
#
cd $RPM_BUILD_DIR/llvm
mkdir build
cp -rf include/ build/include/
cd build
%if %debug
../configure --enable-jit --enable-debug-runtime --enable-debug-symbols --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe,ptx --with-cloog=$RPM_BUILD_ROOT/opt/kernelgen --with-isl=$RPM_BUILD_ROOT/opt/kernelgen
%else
../configure --enable-jit --enable-optimized --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe,ptx --with-cloog=$RPM_BUILD_ROOT/opt/kernelgen --with-isl=$RPM_BUILD_ROOT/opt/kernelgen
%endif
#
# Configure GCC
#
cd $RPM_BUILD_DIR/gcc-4.6.3
mkdir build
cd build/
../configure --prefix=$RPM_BUILD_ROOT/opt/kernelgen --program-prefix=kernelgen- --enable-languages=fortran --with-mpfr-include=/usr/include/ --with-mpfr-lib=/usr/lib64 --with-gmp-include=/usr/include/ --with-gmp-lib=/usr/lib64 --enable-plugin
%endif
#
# Configure KernelGen
#
rm -rf $RPM_BUILD_DIR/kernelgen
cd $RPM_BUILD_DIR
tar -xf $RPM_SOURCE_DIR/kernelgen-r698.tar.gz
cd $RPM_BUILD_DIR/kernelgen
./configure
cd $RPM_BUILD_DIR
#
# Build parts of the system
#
%if %fullrepack
#
# Build NVOPENCC (Open64 compiler with NVIDIA's PTX backend)
#
%if %debug
cd $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa
make
%else
cd $RPM_BUILD_DIR/nvopencc/open64/src/targia3264_nvisa_rel
make
%endif
#
# Build LLVM.
#
cd $RPM_BUILD_DIR/llvm/build
%if %debug
make -j%{njobs} CXXFLAGS=-O0
%else
make -j%{njobs}
%endif
#
# Build original GCC.
#
cd $RPM_BUILD_DIR/gcc-4.6.3/build
%if %debug
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu make -j%{njobs} CFLAGS="-g -O0" CXXFLAGS="-g -O0"
%else
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu  make -j%{njobs}
%endif
#
# Build DragonEgg
#
cd $RPM_BUILD_DIR/dragonegg
GCC=$RPM_BUILD_DIR/gcc-4.6.3/build/gcc/xgcc LLVM_CONFIG=$RPM_BUILD_DIR/llvm/build/Debug+Asserts/bin/llvm-config make clean
%if %debug
CPLUS_INCLUDE_PATH=$RPM_BUILD_DIR/gcc-4.6.3/gcc/:$RPM_BUILD_DIR/gcc-4.6.3/build/gcc/:$RPM_BUILD_DIR/gcc-4.6.3/include/:$RPM_BUILD_DIR/gcc-4.6.3/libcpp/include/ GCC=$RPM_BUILD_DIR/gcc-4.6.3/build/gcc/xgcc LLVM_CONFIG=$RPM_BUILD_DIR/llvm/build/Debug+Asserts/bin/llvm-config make CXXFLAGS="-g -O0 -fPIC"
%else
CPLUS_INCLUDE_PATH=$RPM_BUILD_DIR/gcc-4.6.3/gcc/:$RPM_BUILD_DIR/gcc-4.6.3/build/gcc/:$RPM_BUILD_DIR/gcc-4.6.3/include/:$RPM_BUILD_DIR/gcc-4.6.3/libcpp/include/ GCC=$RPM_BUILD_DIR/gcc-4.6.3/build/gcc/xgcc LLVM_CONFIG=$RPM_BUILD_DIR/llvm/build/Debug+Asserts/bin/llvm-config make
%endif
cd $RPM_BUILD_DIR
patch -p1 <$RPM_SOURCE_DIR/gcc.patch
%endif
#
# Build KernelGen
#
cd $RPM_BUILD_DIR/kernelgen
make src
#
# Build modified GCC.
# Note GCC depends on DragonEgg and plugins from KernelGen,
# thus they both must be built and installed prior to GCC.
#
mkdir -p $RPM_BUILD_ROOT/opt/kernelgen/lib
cp $RPM_BUILD_DIR/dragonegg/dragonegg.so $RPM_BUILD_ROOT/opt/kernelgen/lib/
cp $RPM_BUILD_DIR/kernelgen/src/frontend/libkernelgen-ct.so $RPM_BUILD_ROOT/opt/kernelgen/lib/
cp $RPM_BUILD_DIR/kernelgen/src/frontend/libkernelgen-lt.so $RPM_BUILD_ROOT/opt/kernelgen/lib/
cd $RPM_BUILD_DIR/gcc-4.6.3/build/gcc
%if %debug
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RPM_BUILD_DIR/llvm/build/Debug+Asserts/lib KERNELGEN_PLUGINS_PATH=$RPM_BUILD_ROOT/opt/kernelgen/lib/ LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu make -j%{njobs} CFLAGS="-g -O0" CXXFLAGS="-g -O0"
%else
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RPM_BUILD_DIR/llvm/build/Debug+Asserts/lib KERNELGEN_PLUGINS_PATH=$RPM_BUILD_ROOT/opt/kernelgen/lib/ LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu  make -j%{njobs}
%endif


#
# Install software to the build root.
#
%install
#
# Create directories srtucture.
#
rm -rf $RPM_BUILD_ROOT
mkdir -p $RPM_BUILD_ROOT/opt/kernelgen/bin
mkdir -p $RPM_BUILD_ROOT/opt/kernelgen/%{lib64}/
#
# Reinstall CLooG.
#
cd $RPM_BUILD_DIR/cloog
make install
#
# Install NVOPENCC.
#
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
#
# Install LLVM.
#
cd $RPM_BUILD_DIR/llvm/build
make install
#
# Install DragonEgg
#
cp $RPM_BUILD_DIR/dragonegg/dragonegg.so $RPM_BUILD_ROOT/opt/kernelgen/lib/
#
# Install KernelGen.
#
cd $RPM_BUILD_DIR/kernelgen
ROOT=$RPM_BUILD_ROOT LIB32=lib LIB64=lib make install
#
# Install GCC.
#
cd $RPM_BUILD_DIR/gcc-4.6.3/build
KERNELGEN_PLUGINS_PATH=$RPM_BUILD_ROOT/opt/kernelgen/lib/ LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RPM_BUILD_DIR/dragonegg:$RPM_BUILD_DIR/kernelgen/src/frontend:$RPM_BUILD_DIR/llvm/build/Debug+Asserts/lib LIBRARY_PATH=/usr/lib/x86_64-linux-gnu C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu make install
#
# Clean some unnecessary files.
#
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/cpp.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/cppinternals.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/dir
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/gcc.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/gccinstall.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/gccint.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/gfortran.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/libgomp.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/share/info/libquadmath.info
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/openssl/bn.h


#
# Final cleanup (off for fast repack).
#
%clean
#rm -rf $RPM_BUILD_DIR/cloog
#rm -rf $RPM_BUILD_DIR/scoplib-0.2.0
#rm -rf $RPM_BUILD_DIR/llvm
#rm -rf $RPM_BUILD_DIR/gcc
#rm -rf $RPM_BUILD_DIR/dragonegg
#rm -rf $RPM_BUILD_DIR/kernelgen


#
# Files included into binary distribution.
#
%files
#
# NVOPENCC files.
#
/opt/kernelgen/bin/nvopencc
/opt/kernelgen/lib/be
/opt/kernelgen/lib/gfec
/opt/kernelgen/lib/inline
#
# KernelGen files.
#
/opt/kernelgen/include/kernelgen_interop.h
/opt/kernelgen/include/kernelgen_memory.h
/opt/kernelgen/include/kernelgen_runtime.h
/opt/kernelgen/lib/libkernelgen-ct.so
/opt/kernelgen/lib/libkernelgen-lt.so
/opt/kernelgen/lib/libkernelgen.a
/opt/kernelgen/lib/libasfermi.so
/opt/kernelgen/lib/libdyloader.so
#
# DragonEgg files.
#
/opt/kernelgen/lib/dragonegg.so
#
# CLooG files.
#
/opt/kernelgen/bin/cloog
/opt/kernelgen/include/cloog/block.h
/opt/kernelgen/include/cloog/clast.h
/opt/kernelgen/include/cloog/cloog.h
/opt/kernelgen/include/cloog/constraints.h
/opt/kernelgen/include/cloog/domain.h
/opt/kernelgen/include/cloog/input.h
/opt/kernelgen/include/cloog/int.h
/opt/kernelgen/include/cloog/isl/backend.h
/opt/kernelgen/include/cloog/isl/cloog.h
/opt/kernelgen/include/cloog/isl/constraintset.h
/opt/kernelgen/include/cloog/isl/domain.h
/opt/kernelgen/include/cloog/loop.h
/opt/kernelgen/include/cloog/matrix.h
/opt/kernelgen/include/cloog/matrix/constraintset.h
/opt/kernelgen/include/cloog/names.h
/opt/kernelgen/include/cloog/options.h
/opt/kernelgen/include/cloog/pprint.h
/opt/kernelgen/include/cloog/program.h
/opt/kernelgen/include/cloog/state.h
/opt/kernelgen/include/cloog/statement.h
/opt/kernelgen/include/cloog/stride.h
/opt/kernelgen/include/cloog/union_domain.h
/opt/kernelgen/include/cloog/version.h
/opt/kernelgen/include/isl/aff.h
/opt/kernelgen/include/isl/aff_type.h
/opt/kernelgen/include/isl/arg.h
/opt/kernelgen/include/isl/band.h
/opt/kernelgen/include/isl/blk.h
/opt/kernelgen/include/isl/config.h
/opt/kernelgen/include/isl/constraint.h
/opt/kernelgen/include/isl/ctx.h
/opt/kernelgen/include/isl/dim.h
/opt/kernelgen/include/isl/flow.h
/opt/kernelgen/include/isl/hash.h
/opt/kernelgen/include/isl/id.h
/opt/kernelgen/include/isl/ilp.h
/opt/kernelgen/include/isl/int.h
/opt/kernelgen/include/isl/list.h
/opt/kernelgen/include/isl/local_space.h
/opt/kernelgen/include/isl/lp.h
/opt/kernelgen/include/isl/map.h
/opt/kernelgen/include/isl/map_type.h
/opt/kernelgen/include/isl/mat.h
/opt/kernelgen/include/isl/multi.h
/opt/kernelgen/include/isl/obj.h
/opt/kernelgen/include/isl/options.h
/opt/kernelgen/include/isl/point.h
/opt/kernelgen/include/isl/polynomial.h
/opt/kernelgen/include/isl/polynomial_type.h
/opt/kernelgen/include/isl/printer.h
/opt/kernelgen/include/isl/schedule.h
/opt/kernelgen/include/isl/seq.h
/opt/kernelgen/include/isl/set.h
/opt/kernelgen/include/isl/set_type.h
/opt/kernelgen/include/isl/space.h
/opt/kernelgen/include/isl/stdint.h
/opt/kernelgen/include/isl/stream.h
/opt/kernelgen/include/isl/union_map.h
/opt/kernelgen/include/isl/union_set.h
/opt/kernelgen/include/isl/vec.h
/opt/kernelgen/include/isl/version.h
/opt/kernelgen/include/isl/vertices.h
/opt/kernelgen/lib/libcloog-isl.a
/opt/kernelgen/lib/libcloog-isl.la
/opt/kernelgen/lib/libcloog-isl.so
/opt/kernelgen/lib/libcloog-isl.so.3
/opt/kernelgen/lib/libcloog-isl.so.3.0.0
/opt/kernelgen/lib/libisl.a
/opt/kernelgen/lib/libisl.la
/opt/kernelgen/lib/libisl.so
/opt/kernelgen/lib/libisl.so.9
/opt/kernelgen/lib/libisl.so.9.0.0
/opt/kernelgen/lib/libisl.so.9.0.0-gdb.py
/opt/kernelgen/lib/pkgconfig/cloog-isl.pc
/opt/kernelgen/lib/pkgconfig/isl.pc
#
# LLVM files.
#
/opt/kernelgen/bin/bugpoint
/opt/kernelgen/bin/llc
/opt/kernelgen/bin/lli
/opt/kernelgen/bin/llvm-ar
/opt/kernelgen/bin/llvm-as
/opt/kernelgen/bin/llvm-bcanalyzer
/opt/kernelgen/bin/llvm-config
/opt/kernelgen/bin/llvm-cov
/opt/kernelgen/bin/llvm-diff
/opt/kernelgen/bin/llvm-dis
/opt/kernelgen/bin/llvm-dwarfdump
/opt/kernelgen/bin/llvm-extract
/opt/kernelgen/bin/llvm-ld
/opt/kernelgen/bin/llvm-link
/opt/kernelgen/bin/llvm-mc
/opt/kernelgen/bin/llvm-nm
/opt/kernelgen/bin/llvm-objdump
/opt/kernelgen/bin/llvm-prof
/opt/kernelgen/bin/llvm-ranlib
/opt/kernelgen/bin/llvm-rtdyld
/opt/kernelgen/bin/llvm-size
/opt/kernelgen/bin/llvm-stub
/opt/kernelgen/bin/llvm-tblgen
/opt/kernelgen/bin/macho-dump
/opt/kernelgen/bin/opt
/opt/kernelgen/docs/llvm/html.tar.gz
/opt/kernelgen/docs/llvm/html/AliasAnalysis.html
/opt/kernelgen/docs/llvm/html/Atomics.html
/opt/kernelgen/docs/llvm/html/BitCodeFormat.html
/opt/kernelgen/docs/llvm/html/BranchWeightMetadata.html
/opt/kernelgen/docs/llvm/html/Bugpoint.html
/opt/kernelgen/docs/llvm/html/CFEBuildInstrs.html
/opt/kernelgen/docs/llvm/html/CMake.html
/opt/kernelgen/docs/llvm/html/CodeGenerator.html
/opt/kernelgen/docs/llvm/html/CodingStandards.html
/opt/kernelgen/docs/llvm/html/CommandGuide/FileCheck.html
/opt/kernelgen/docs/llvm/html/CommandGuide/bugpoint.html
/opt/kernelgen/docs/llvm/html/CommandGuide/index.html
/opt/kernelgen/docs/llvm/html/CommandGuide/lit.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llc.html
/opt/kernelgen/docs/llvm/html/CommandGuide/lli.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ar.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-as.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-bcanalyzer.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-build.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-config.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-cov.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-diff.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-dis.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-extract.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ld.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-link.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-nm.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-prof.html
/opt/kernelgen/docs/llvm/html/CommandGuide/llvm-ranlib.html
/opt/kernelgen/docs/llvm/html/CommandGuide/manpage.css
/opt/kernelgen/docs/llvm/html/CommandGuide/opt.html
/opt/kernelgen/docs/llvm/html/CommandGuide/tblgen.html
/opt/kernelgen/docs/llvm/html/CommandLine.html
/opt/kernelgen/docs/llvm/html/CompilerWriterInfo.html
/opt/kernelgen/docs/llvm/html/DebuggingJITedCode.html
/opt/kernelgen/docs/llvm/html/DeveloperPolicy.html
/opt/kernelgen/docs/llvm/html/ExceptionHandling.html
/opt/kernelgen/docs/llvm/html/ExtendingLLVM.html
/opt/kernelgen/docs/llvm/html/FAQ.html
/opt/kernelgen/docs/llvm/html/GCCFEBuildInstrs.html
/opt/kernelgen/docs/llvm/html/GarbageCollection.html
/opt/kernelgen/docs/llvm/html/GetElementPtr.html
/opt/kernelgen/docs/llvm/html/GettingStarted.html
/opt/kernelgen/docs/llvm/html/GettingStartedVS.html
/opt/kernelgen/docs/llvm/html/GoldPlugin.html
/opt/kernelgen/docs/llvm/html/HowToAddABuilder.html
/opt/kernelgen/docs/llvm/html/HowToReleaseLLVM.html
/opt/kernelgen/docs/llvm/html/HowToSubmitABug.html
/opt/kernelgen/docs/llvm/html/LLVMBuild.html
/opt/kernelgen/docs/llvm/html/LangRef.html
/opt/kernelgen/docs/llvm/html/Lexicon.html
/opt/kernelgen/docs/llvm/html/LinkTimeOptimization.html
/opt/kernelgen/docs/llvm/html/MakefileGuide.html
/opt/kernelgen/docs/llvm/html/Packaging.html
/opt/kernelgen/docs/llvm/html/Passes.html
/opt/kernelgen/docs/llvm/html/ProgrammersManual.html
/opt/kernelgen/docs/llvm/html/Projects.html
/opt/kernelgen/docs/llvm/html/ReleaseNotes.html
/opt/kernelgen/docs/llvm/html/SegmentedStacks.html
/opt/kernelgen/docs/llvm/html/SourceLevelDebugging.html
/opt/kernelgen/docs/llvm/html/SystemLibrary.html
/opt/kernelgen/docs/llvm/html/TableGenFundamentals.html
/opt/kernelgen/docs/llvm/html/TestingGuide.html
/opt/kernelgen/docs/llvm/html/WritingAnLLVMBackend.html
/opt/kernelgen/docs/llvm/html/WritingAnLLVMPass.html
/opt/kernelgen/docs/llvm/html/doxygen.css
/opt/kernelgen/docs/llvm/html/img/Debugging.gif
/opt/kernelgen/docs/llvm/html/img/libdeps.gif
/opt/kernelgen/docs/llvm/html/img/lines.gif
/opt/kernelgen/docs/llvm/html/img/objdeps.gif
/opt/kernelgen/docs/llvm/html/img/venusflytrap.jpg
/opt/kernelgen/docs/llvm/html/index.html
/opt/kernelgen/docs/llvm/html/llvm.css
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl1.html
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl2.html
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl3.html
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl4.html
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl5-cfg.png
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl5.html
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl6.html
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl7.html
/opt/kernelgen/docs/llvm/html/tutorial/LangImpl8.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl1.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl2.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl3.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl4.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl5.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl6.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl7.html
/opt/kernelgen/docs/llvm/html/tutorial/OCamlLangImpl8.html
/opt/kernelgen/docs/llvm/html/tutorial/index.html
/opt/kernelgen/docs/llvm/ps/FileCheck.ps
/opt/kernelgen/docs/llvm/ps/bugpoint.ps
/opt/kernelgen/docs/llvm/ps/lit.ps
/opt/kernelgen/docs/llvm/ps/llc.ps
/opt/kernelgen/docs/llvm/ps/lli.ps
/opt/kernelgen/docs/llvm/ps/llvm-ar.ps
/opt/kernelgen/docs/llvm/ps/llvm-as.ps
/opt/kernelgen/docs/llvm/ps/llvm-bcanalyzer.ps
/opt/kernelgen/docs/llvm/ps/llvm-build.ps
/opt/kernelgen/docs/llvm/ps/llvm-config.ps
/opt/kernelgen/docs/llvm/ps/llvm-cov.ps
/opt/kernelgen/docs/llvm/ps/llvm-diff.ps
/opt/kernelgen/docs/llvm/ps/llvm-dis.ps
/opt/kernelgen/docs/llvm/ps/llvm-extract.ps
/opt/kernelgen/docs/llvm/ps/llvm-ld.ps
/opt/kernelgen/docs/llvm/ps/llvm-link.ps
/opt/kernelgen/docs/llvm/ps/llvm-nm.ps
/opt/kernelgen/docs/llvm/ps/llvm-prof.ps
/opt/kernelgen/docs/llvm/ps/llvm-ranlib.ps
/opt/kernelgen/docs/llvm/ps/opt.ps
/opt/kernelgen/docs/llvm/ps/tblgen.ps
/opt/kernelgen/include/llvm-c/Analysis.h
/opt/kernelgen/include/llvm-c/BitReader.h
/opt/kernelgen/include/llvm-c/BitWriter.h
/opt/kernelgen/include/llvm-c/Core.h
/opt/kernelgen/include/llvm-c/Disassembler.h
/opt/kernelgen/include/llvm-c/EnhancedDisassembly.h
/opt/kernelgen/include/llvm-c/ExecutionEngine.h
/opt/kernelgen/include/llvm-c/Initialization.h
/opt/kernelgen/include/llvm-c/LinkTimeOptimizer.h
/opt/kernelgen/include/llvm-c/Object.h
/opt/kernelgen/include/llvm-c/Target.h
/opt/kernelgen/include/llvm-c/Transforms/IPO.h
/opt/kernelgen/include/llvm-c/Transforms/PassManagerBuilder.h
/opt/kernelgen/include/llvm-c/Transforms/Scalar.h
/opt/kernelgen/include/llvm-c/Transforms/Vectorize.h
/opt/kernelgen/include/llvm-c/lto.h
/opt/kernelgen/include/llvm/ADT/APFloat.h
/opt/kernelgen/include/llvm/ADT/APInt.h
/opt/kernelgen/include/llvm/ADT/APSInt.h
/opt/kernelgen/include/llvm/ADT/ArrayRef.h
/opt/kernelgen/include/llvm/ADT/BitVector.h
/opt/kernelgen/include/llvm/ADT/DAGDeltaAlgorithm.h
/opt/kernelgen/include/llvm/ADT/DeltaAlgorithm.h
/opt/kernelgen/include/llvm/ADT/DenseMap.h
/opt/kernelgen/include/llvm/ADT/DenseMapInfo.h
/opt/kernelgen/include/llvm/ADT/DenseSet.h
/opt/kernelgen/include/llvm/ADT/DepthFirstIterator.h
/opt/kernelgen/include/llvm/ADT/EquivalenceClasses.h
/opt/kernelgen/include/llvm/ADT/FoldingSet.h
/opt/kernelgen/include/llvm/ADT/GraphTraits.h
/opt/kernelgen/include/llvm/ADT/Hashing.h
/opt/kernelgen/include/llvm/ADT/ImmutableIntervalMap.h
/opt/kernelgen/include/llvm/ADT/ImmutableList.h
/opt/kernelgen/include/llvm/ADT/ImmutableMap.h
/opt/kernelgen/include/llvm/ADT/ImmutableSet.h
/opt/kernelgen/include/llvm/ADT/InMemoryStruct.h
/opt/kernelgen/include/llvm/ADT/IndexedMap.h
/opt/kernelgen/include/llvm/ADT/IntEqClasses.h
/opt/kernelgen/include/llvm/ADT/IntervalMap.h
/opt/kernelgen/include/llvm/ADT/IntrusiveRefCntPtr.h
/opt/kernelgen/include/llvm/ADT/NullablePtr.h
/opt/kernelgen/include/llvm/ADT/Optional.h
/opt/kernelgen/include/llvm/ADT/OwningPtr.h
/opt/kernelgen/include/llvm/ADT/PackedVector.h
/opt/kernelgen/include/llvm/ADT/PointerIntPair.h
/opt/kernelgen/include/llvm/ADT/PointerUnion.h
/opt/kernelgen/include/llvm/ADT/PostOrderIterator.h
/opt/kernelgen/include/llvm/ADT/PriorityQueue.h
/opt/kernelgen/include/llvm/ADT/SCCIterator.h
/opt/kernelgen/include/llvm/ADT/STLExtras.h
/opt/kernelgen/include/llvm/ADT/ScopedHashTable.h
/opt/kernelgen/include/llvm/ADT/SetOperations.h
/opt/kernelgen/include/llvm/ADT/SetVector.h
/opt/kernelgen/include/llvm/ADT/SmallBitVector.h
/opt/kernelgen/include/llvm/ADT/SmallPtrSet.h
/opt/kernelgen/include/llvm/ADT/SmallSet.h
/opt/kernelgen/include/llvm/ADT/SmallString.h
/opt/kernelgen/include/llvm/ADT/SmallVector.h
/opt/kernelgen/include/llvm/ADT/SparseBitVector.h
/opt/kernelgen/include/llvm/ADT/Statistic.h
/opt/kernelgen/include/llvm/ADT/StringExtras.h
/opt/kernelgen/include/llvm/ADT/StringMap.h
/opt/kernelgen/include/llvm/ADT/StringRef.h
/opt/kernelgen/include/llvm/ADT/StringSet.h
/opt/kernelgen/include/llvm/ADT/StringSwitch.h
/opt/kernelgen/include/llvm/ADT/TinyPtrVector.h
/opt/kernelgen/include/llvm/ADT/Trie.h
/opt/kernelgen/include/llvm/ADT/Triple.h
/opt/kernelgen/include/llvm/ADT/Twine.h
/opt/kernelgen/include/llvm/ADT/UniqueVector.h
/opt/kernelgen/include/llvm/ADT/ValueMap.h
/opt/kernelgen/include/llvm/ADT/VariadicFunction.h
/opt/kernelgen/include/llvm/ADT/edit_distance.h
/opt/kernelgen/include/llvm/ADT/ilist.h
/opt/kernelgen/include/llvm/ADT/ilist_node.h
/opt/kernelgen/include/llvm/Analysis/AliasAnalysis.h
/opt/kernelgen/include/llvm/Analysis/AliasSetTracker.h
/opt/kernelgen/include/llvm/Analysis/BlockFrequencyImpl.h
/opt/kernelgen/include/llvm/Analysis/BlockFrequencyInfo.h
/opt/kernelgen/include/llvm/Analysis/BranchProbabilityInfo.h
/opt/kernelgen/include/llvm/Analysis/CFGPrinter.h
/opt/kernelgen/include/llvm/Analysis/CallGraph.h
/opt/kernelgen/include/llvm/Analysis/CaptureTracking.h
/opt/kernelgen/include/llvm/Analysis/CodeMetrics.h
/opt/kernelgen/include/llvm/Analysis/ConstantFolding.h
/opt/kernelgen/include/llvm/Analysis/ConstantsScanner.h
/opt/kernelgen/include/llvm/Analysis/DIBuilder.h
/opt/kernelgen/include/llvm/Analysis/DOTGraphTraitsPass.h
/opt/kernelgen/include/llvm/Analysis/DebugInfo.h
/opt/kernelgen/include/llvm/Analysis/DomPrinter.h
/opt/kernelgen/include/llvm/Analysis/DominanceFrontier.h
/opt/kernelgen/include/llvm/Analysis/DominatorInternals.h
/opt/kernelgen/include/llvm/Analysis/Dominators.h
/opt/kernelgen/include/llvm/Analysis/FindUsedTypes.h
/opt/kernelgen/include/llvm/Analysis/IVUsers.h
/opt/kernelgen/include/llvm/Analysis/InlineCost.h
/opt/kernelgen/include/llvm/Analysis/InstructionSimplify.h
/opt/kernelgen/include/llvm/Analysis/Interval.h
/opt/kernelgen/include/llvm/Analysis/IntervalIterator.h
/opt/kernelgen/include/llvm/Analysis/IntervalPartition.h
/opt/kernelgen/include/llvm/Analysis/LazyValueInfo.h
/opt/kernelgen/include/llvm/Analysis/LibCallAliasAnalysis.h
/opt/kernelgen/include/llvm/Analysis/LibCallSemantics.h
/opt/kernelgen/include/llvm/Analysis/Lint.h
/opt/kernelgen/include/llvm/Analysis/Loads.h
/opt/kernelgen/include/llvm/Analysis/LoopDependenceAnalysis.h
/opt/kernelgen/include/llvm/Analysis/LoopInfo.h
/opt/kernelgen/include/llvm/Analysis/LoopIterator.h
/opt/kernelgen/include/llvm/Analysis/LoopPass.h
/opt/kernelgen/include/llvm/Analysis/MemoryBuiltins.h
/opt/kernelgen/include/llvm/Analysis/MemoryDependenceAnalysis.h
/opt/kernelgen/include/llvm/Analysis/PHITransAddr.h
/opt/kernelgen/include/llvm/Analysis/Passes.h
/opt/kernelgen/include/llvm/Analysis/PathNumbering.h
/opt/kernelgen/include/llvm/Analysis/PathProfileInfo.h
/opt/kernelgen/include/llvm/Analysis/PostDominators.h
/opt/kernelgen/include/llvm/Analysis/ProfileInfo.h
/opt/kernelgen/include/llvm/Analysis/ProfileInfoLoader.h
/opt/kernelgen/include/llvm/Analysis/ProfileInfoTypes.h
/opt/kernelgen/include/llvm/Analysis/RegionInfo.h
/opt/kernelgen/include/llvm/Analysis/RegionIterator.h
/opt/kernelgen/include/llvm/Analysis/RegionPass.h
/opt/kernelgen/include/llvm/Analysis/RegionPrinter.h
/opt/kernelgen/include/llvm/Analysis/ScalarEvolution.h
/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionExpander.h
/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionExpressions.h
/opt/kernelgen/include/llvm/Analysis/ScalarEvolutionNormalization.h
/opt/kernelgen/include/llvm/Analysis/SparsePropagation.h
/opt/kernelgen/include/llvm/Analysis/Trace.h
/opt/kernelgen/include/llvm/Analysis/ValueTracking.h
/opt/kernelgen/include/llvm/Analysis/Verifier.h
/opt/kernelgen/include/llvm/Argument.h
/opt/kernelgen/include/llvm/Assembly/AssemblyAnnotationWriter.h
/opt/kernelgen/include/llvm/Assembly/Parser.h
/opt/kernelgen/include/llvm/Assembly/PrintModulePass.h
/opt/kernelgen/include/llvm/Assembly/Writer.h
/opt/kernelgen/include/llvm/Attributes.h
/opt/kernelgen/include/llvm/AutoUpgrade.h
/opt/kernelgen/include/llvm/BasicBlock.h
/opt/kernelgen/include/llvm/Bitcode/Archive.h
/opt/kernelgen/include/llvm/Bitcode/BitCodes.h
/opt/kernelgen/include/llvm/Bitcode/BitstreamReader.h
/opt/kernelgen/include/llvm/Bitcode/BitstreamWriter.h
/opt/kernelgen/include/llvm/Bitcode/LLVMBitCodes.h
/opt/kernelgen/include/llvm/Bitcode/ReaderWriter.h
/opt/kernelgen/include/llvm/CallGraphSCCPass.h
/opt/kernelgen/include/llvm/CallingConv.h
/opt/kernelgen/include/llvm/CodeGen/Analysis.h
/opt/kernelgen/include/llvm/CodeGen/AsmPrinter.h
/opt/kernelgen/include/llvm/CodeGen/CalcSpillWeights.h
/opt/kernelgen/include/llvm/CodeGen/CallingConvLower.h
/opt/kernelgen/include/llvm/CodeGen/DFAPacketizer.h
/opt/kernelgen/include/llvm/CodeGen/EdgeBundles.h
/opt/kernelgen/include/llvm/CodeGen/FastISel.h
/opt/kernelgen/include/llvm/CodeGen/FunctionLoweringInfo.h
/opt/kernelgen/include/llvm/CodeGen/GCMetadata.h
/opt/kernelgen/include/llvm/CodeGen/GCMetadataPrinter.h
/opt/kernelgen/include/llvm/CodeGen/GCStrategy.h
/opt/kernelgen/include/llvm/CodeGen/GCs.h
/opt/kernelgen/include/llvm/CodeGen/ISDOpcodes.h
/opt/kernelgen/include/llvm/CodeGen/IntrinsicLowering.h
/opt/kernelgen/include/llvm/CodeGen/JITCodeEmitter.h
/opt/kernelgen/include/llvm/CodeGen/LatencyPriorityQueue.h
/opt/kernelgen/include/llvm/CodeGen/LexicalScopes.h
/opt/kernelgen/include/llvm/CodeGen/LinkAllAsmWriterComponents.h
/opt/kernelgen/include/llvm/CodeGen/LinkAllCodegenComponents.h
/opt/kernelgen/include/llvm/CodeGen/LiveInterval.h
/opt/kernelgen/include/llvm/CodeGen/LiveIntervalAnalysis.h
/opt/kernelgen/include/llvm/CodeGen/LiveStackAnalysis.h
/opt/kernelgen/include/llvm/CodeGen/LiveVariables.h
/opt/kernelgen/include/llvm/CodeGen/MachORelocation.h
/opt/kernelgen/include/llvm/CodeGen/MachineBasicBlock.h
/opt/kernelgen/include/llvm/CodeGen/MachineBlockFrequencyInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineBranchProbabilityInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineCodeEmitter.h
/opt/kernelgen/include/llvm/CodeGen/MachineCodeInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineConstantPool.h
/opt/kernelgen/include/llvm/CodeGen/MachineDominators.h
/opt/kernelgen/include/llvm/CodeGen/MachineFrameInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineFunction.h
/opt/kernelgen/include/llvm/CodeGen/MachineFunctionAnalysis.h
/opt/kernelgen/include/llvm/CodeGen/MachineFunctionPass.h
/opt/kernelgen/include/llvm/CodeGen/MachineInstr.h
/opt/kernelgen/include/llvm/CodeGen/MachineInstrBuilder.h
/opt/kernelgen/include/llvm/CodeGen/MachineInstrBundle.h
/opt/kernelgen/include/llvm/CodeGen/MachineJumpTableInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineLoopInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineLoopRanges.h
/opt/kernelgen/include/llvm/CodeGen/MachineMemOperand.h
/opt/kernelgen/include/llvm/CodeGen/MachineModuleInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineModuleInfoImpls.h
/opt/kernelgen/include/llvm/CodeGen/MachineOperand.h
/opt/kernelgen/include/llvm/CodeGen/MachinePassRegistry.h
/opt/kernelgen/include/llvm/CodeGen/MachineRegisterInfo.h
/opt/kernelgen/include/llvm/CodeGen/MachineRelocation.h
/opt/kernelgen/include/llvm/CodeGen/MachineSSAUpdater.h
/opt/kernelgen/include/llvm/CodeGen/PBQP/Graph.h
/opt/kernelgen/include/llvm/CodeGen/PBQP/HeuristicBase.h
/opt/kernelgen/include/llvm/CodeGen/PBQP/HeuristicSolver.h
/opt/kernelgen/include/llvm/CodeGen/PBQP/Heuristics/Briggs.h
/opt/kernelgen/include/llvm/CodeGen/PBQP/Math.h
/opt/kernelgen/include/llvm/CodeGen/PBQP/Solution.h
/opt/kernelgen/include/llvm/CodeGen/Passes.h
/opt/kernelgen/include/llvm/CodeGen/ProcessImplicitDefs.h
/opt/kernelgen/include/llvm/CodeGen/PseudoSourceValue.h
/opt/kernelgen/include/llvm/CodeGen/RegAllocPBQP.h
/opt/kernelgen/include/llvm/CodeGen/RegAllocRegistry.h
/opt/kernelgen/include/llvm/CodeGen/RegisterScavenging.h
/opt/kernelgen/include/llvm/CodeGen/ResourcePriorityQueue.h
/opt/kernelgen/include/llvm/CodeGen/RuntimeLibcalls.h
/opt/kernelgen/include/llvm/CodeGen/ScheduleDAG.h
/opt/kernelgen/include/llvm/CodeGen/ScheduleHazardRecognizer.h
/opt/kernelgen/include/llvm/CodeGen/SchedulerRegistry.h
/opt/kernelgen/include/llvm/CodeGen/ScoreboardHazardRecognizer.h
/opt/kernelgen/include/llvm/CodeGen/SelectionDAG.h
/opt/kernelgen/include/llvm/CodeGen/SelectionDAGISel.h
/opt/kernelgen/include/llvm/CodeGen/SelectionDAGNodes.h
/opt/kernelgen/include/llvm/CodeGen/SlotIndexes.h
/opt/kernelgen/include/llvm/CodeGen/TargetLoweringObjectFileImpl.h
/opt/kernelgen/include/llvm/CodeGen/ValueTypes.h
/opt/kernelgen/include/llvm/CodeGen/ValueTypes.td
/opt/kernelgen/include/llvm/Config/AsmParsers.def
/opt/kernelgen/include/llvm/Config/AsmPrinters.def
/opt/kernelgen/include/llvm/Config/Disassemblers.def
/opt/kernelgen/include/llvm/Config/Targets.def
/opt/kernelgen/include/llvm/Config/config.h
/opt/kernelgen/include/llvm/Config/llvm-config.h
/opt/kernelgen/include/llvm/Constant.h
/opt/kernelgen/include/llvm/Constants.h
/opt/kernelgen/include/llvm/DebugInfo/DIContext.h
/opt/kernelgen/include/llvm/DebugInfoProbe.h
/opt/kernelgen/include/llvm/DefaultPasses.h
/opt/kernelgen/include/llvm/DerivedTypes.h
/opt/kernelgen/include/llvm/ExecutionEngine/ExecutionEngine.h
/opt/kernelgen/include/llvm/ExecutionEngine/GenericValue.h
/opt/kernelgen/include/llvm/ExecutionEngine/Interpreter.h
/opt/kernelgen/include/llvm/ExecutionEngine/JIT.h
/opt/kernelgen/include/llvm/ExecutionEngine/JITEventListener.h
/opt/kernelgen/include/llvm/ExecutionEngine/JITMemoryManager.h
/opt/kernelgen/include/llvm/ExecutionEngine/MCJIT.h
/opt/kernelgen/include/llvm/ExecutionEngine/RuntimeDyld.h
/opt/kernelgen/include/llvm/Function.h
/opt/kernelgen/include/llvm/GVMaterializer.h
/opt/kernelgen/include/llvm/GlobalAlias.h
/opt/kernelgen/include/llvm/GlobalValue.h
/opt/kernelgen/include/llvm/GlobalVariable.h
/opt/kernelgen/include/llvm/InitializePasses.h
/opt/kernelgen/include/llvm/InlineAsm.h
/opt/kernelgen/include/llvm/InstrTypes.h
/opt/kernelgen/include/llvm/Instruction.def
/opt/kernelgen/include/llvm/Instruction.h
/opt/kernelgen/include/llvm/Instructions.h
/opt/kernelgen/include/llvm/IntrinsicInst.h
/opt/kernelgen/include/llvm/Intrinsics.gen
/opt/kernelgen/include/llvm/Intrinsics.h
/opt/kernelgen/include/llvm/Intrinsics.td
/opt/kernelgen/include/llvm/IntrinsicsARM.td
/opt/kernelgen/include/llvm/IntrinsicsCellSPU.td
/opt/kernelgen/include/llvm/IntrinsicsHexagon.td
/opt/kernelgen/include/llvm/IntrinsicsPTX.td
/opt/kernelgen/include/llvm/IntrinsicsPowerPC.td
/opt/kernelgen/include/llvm/IntrinsicsX86.td
/opt/kernelgen/include/llvm/IntrinsicsXCore.td
/opt/kernelgen/include/llvm/LLVMContext.h
/opt/kernelgen/include/llvm/LinkAllPasses.h
/opt/kernelgen/include/llvm/LinkAllVMCore.h
/opt/kernelgen/include/llvm/Linker.h
/opt/kernelgen/include/llvm/MC/EDInstInfo.h
/opt/kernelgen/include/llvm/MC/MCAsmBackend.h
/opt/kernelgen/include/llvm/MC/MCAsmInfo.h
/opt/kernelgen/include/llvm/MC/MCAsmInfoCOFF.h
/opt/kernelgen/include/llvm/MC/MCAsmInfoDarwin.h
/opt/kernelgen/include/llvm/MC/MCAsmLayout.h
/opt/kernelgen/include/llvm/MC/MCAssembler.h
/opt/kernelgen/include/llvm/MC/MCAtom.h
/opt/kernelgen/include/llvm/MC/MCCodeEmitter.h
/opt/kernelgen/include/llvm/MC/MCCodeGenInfo.h
/opt/kernelgen/include/llvm/MC/MCContext.h
/opt/kernelgen/include/llvm/MC/MCDirectives.h
/opt/kernelgen/include/llvm/MC/MCDisassembler.h
/opt/kernelgen/include/llvm/MC/MCDwarf.h
/opt/kernelgen/include/llvm/MC/MCELFObjectWriter.h
/opt/kernelgen/include/llvm/MC/MCELFSymbolFlags.h
/opt/kernelgen/include/llvm/MC/MCExpr.h
/opt/kernelgen/include/llvm/MC/MCFixup.h
/opt/kernelgen/include/llvm/MC/MCFixupKindInfo.h
/opt/kernelgen/include/llvm/MC/MCInst.h
/opt/kernelgen/include/llvm/MC/MCInstPrinter.h
/opt/kernelgen/include/llvm/MC/MCInstrAnalysis.h
/opt/kernelgen/include/llvm/MC/MCInstrDesc.h
/opt/kernelgen/include/llvm/MC/MCInstrInfo.h
/opt/kernelgen/include/llvm/MC/MCInstrItineraries.h
/opt/kernelgen/include/llvm/MC/MCLabel.h
/opt/kernelgen/include/llvm/MC/MCMachOSymbolFlags.h
/opt/kernelgen/include/llvm/MC/MCMachObjectWriter.h
/opt/kernelgen/include/llvm/MC/MCModule.h
/opt/kernelgen/include/llvm/MC/MCObjectFileInfo.h
/opt/kernelgen/include/llvm/MC/MCObjectStreamer.h
/opt/kernelgen/include/llvm/MC/MCObjectWriter.h
/opt/kernelgen/include/llvm/MC/MCParser/AsmCond.h
/opt/kernelgen/include/llvm/MC/MCParser/AsmLexer.h
/opt/kernelgen/include/llvm/MC/MCParser/MCAsmLexer.h
/opt/kernelgen/include/llvm/MC/MCParser/MCAsmParser.h
/opt/kernelgen/include/llvm/MC/MCParser/MCAsmParserExtension.h
/opt/kernelgen/include/llvm/MC/MCParser/MCParsedAsmOperand.h
/opt/kernelgen/include/llvm/MC/MCRegisterInfo.h
/opt/kernelgen/include/llvm/MC/MCSection.h
/opt/kernelgen/include/llvm/MC/MCSectionCOFF.h
/opt/kernelgen/include/llvm/MC/MCSectionELF.h
/opt/kernelgen/include/llvm/MC/MCSectionMachO.h
/opt/kernelgen/include/llvm/MC/MCStreamer.h
/opt/kernelgen/include/llvm/MC/MCSubtargetInfo.h
/opt/kernelgen/include/llvm/MC/MCSymbol.h
/opt/kernelgen/include/llvm/MC/MCTargetAsmLexer.h
/opt/kernelgen/include/llvm/MC/MCTargetAsmParser.h
/opt/kernelgen/include/llvm/MC/MCValue.h
/opt/kernelgen/include/llvm/MC/MCWin64EH.h
/opt/kernelgen/include/llvm/MC/MCWinCOFFObjectWriter.h
/opt/kernelgen/include/llvm/MC/MachineLocation.h
/opt/kernelgen/include/llvm/MC/SectionKind.h
/opt/kernelgen/include/llvm/MC/SubtargetFeature.h
/opt/kernelgen/include/llvm/Metadata.h
/opt/kernelgen/include/llvm/Module.h
/opt/kernelgen/include/llvm/Object/Archive.h
/opt/kernelgen/include/llvm/Object/Binary.h
/opt/kernelgen/include/llvm/Object/COFF.h
/opt/kernelgen/include/llvm/Object/ELF.h
/opt/kernelgen/include/llvm/Object/Error.h
/opt/kernelgen/include/llvm/Object/MachO.h
/opt/kernelgen/include/llvm/Object/MachOFormat.h
/opt/kernelgen/include/llvm/Object/MachOObject.h
/opt/kernelgen/include/llvm/Object/ObjectFile.h
/opt/kernelgen/include/llvm/OperandTraits.h
/opt/kernelgen/include/llvm/Operator.h
/opt/kernelgen/include/llvm/Pass.h
/opt/kernelgen/include/llvm/PassAnalysisSupport.h
/opt/kernelgen/include/llvm/PassManager.h
/opt/kernelgen/include/llvm/PassManagers.h
/opt/kernelgen/include/llvm/PassRegistry.h
/opt/kernelgen/include/llvm/PassSupport.h
/opt/kernelgen/include/llvm/Support/AIXDataTypesFix.h
/opt/kernelgen/include/llvm/Support/AlignOf.h
/opt/kernelgen/include/llvm/Support/Allocator.h
/opt/kernelgen/include/llvm/Support/Atomic.h
/opt/kernelgen/include/llvm/Support/BlockFrequency.h
/opt/kernelgen/include/llvm/Support/BranchProbability.h
/opt/kernelgen/include/llvm/Support/CFG.h
/opt/kernelgen/include/llvm/Support/COFF.h
/opt/kernelgen/include/llvm/Support/CallSite.h
/opt/kernelgen/include/llvm/Support/Capacity.h
/opt/kernelgen/include/llvm/Support/Casting.h
/opt/kernelgen/include/llvm/Support/CodeGen.h
/opt/kernelgen/include/llvm/Support/CommandLine.h
/opt/kernelgen/include/llvm/Support/Compiler.h
/opt/kernelgen/include/llvm/Support/ConstantFolder.h
/opt/kernelgen/include/llvm/Support/ConstantRange.h
/opt/kernelgen/include/llvm/Support/CrashRecoveryContext.h
/opt/kernelgen/include/llvm/Support/DOTGraphTraits.h
/opt/kernelgen/include/llvm/Support/DataExtractor.h
/opt/kernelgen/include/llvm/Support/DataFlow.h
/opt/kernelgen/include/llvm/Support/DataStream.h
/opt/kernelgen/include/llvm/Support/DataTypes.h
/opt/kernelgen/include/llvm/Support/Debug.h
/opt/kernelgen/include/llvm/Support/DebugLoc.h
/opt/kernelgen/include/llvm/Support/Disassembler.h
/opt/kernelgen/include/llvm/Support/Dwarf.h
/opt/kernelgen/include/llvm/Support/DynamicLibrary.h
/opt/kernelgen/include/llvm/Support/ELF.h
/opt/kernelgen/include/llvm/Support/Endian.h
/opt/kernelgen/include/llvm/Support/Errno.h
/opt/kernelgen/include/llvm/Support/ErrorHandling.h
/opt/kernelgen/include/llvm/Support/FEnv.h
/opt/kernelgen/include/llvm/Support/FileSystem.h
/opt/kernelgen/include/llvm/Support/FileUtilities.h
/opt/kernelgen/include/llvm/Support/Format.h
/opt/kernelgen/include/llvm/Support/FormattedStream.h
/opt/kernelgen/include/llvm/Support/GCOV.h
/opt/kernelgen/include/llvm/Support/GetElementPtrTypeIterator.h
/opt/kernelgen/include/llvm/Support/GraphWriter.h
/opt/kernelgen/include/llvm/Support/Host.h
/opt/kernelgen/include/llvm/Support/IRBuilder.h
/opt/kernelgen/include/llvm/Support/IRReader.h
/opt/kernelgen/include/llvm/Support/IncludeFile.h
/opt/kernelgen/include/llvm/Support/InstIterator.h
/opt/kernelgen/include/llvm/Support/InstVisitor.h
/opt/kernelgen/include/llvm/Support/JSONParser.h
/opt/kernelgen/include/llvm/Support/LICENSE.TXT
/opt/kernelgen/include/llvm/Support/LeakDetector.h
/opt/kernelgen/include/llvm/Support/LockFileManager.h
/opt/kernelgen/include/llvm/Support/MachO.h
/opt/kernelgen/include/llvm/Support/ManagedStatic.h
/opt/kernelgen/include/llvm/Support/MathExtras.h
/opt/kernelgen/include/llvm/Support/Memory.h
/opt/kernelgen/include/llvm/Support/MemoryBuffer.h
/opt/kernelgen/include/llvm/Support/MemoryObject.h
/opt/kernelgen/include/llvm/Support/Mutex.h
/opt/kernelgen/include/llvm/Support/MutexGuard.h
/opt/kernelgen/include/llvm/Support/NoFolder.h
/opt/kernelgen/include/llvm/Support/OutputBuffer.h
/opt/kernelgen/include/llvm/Support/PassNameParser.h
/opt/kernelgen/include/llvm/Support/Path.h
/opt/kernelgen/include/llvm/Support/PathV1.h
/opt/kernelgen/include/llvm/Support/PathV2.h
/opt/kernelgen/include/llvm/Support/PatternMatch.h
/opt/kernelgen/include/llvm/Support/PluginLoader.h
/opt/kernelgen/include/llvm/Support/PointerLikeTypeTraits.h
/opt/kernelgen/include/llvm/Support/PredIteratorCache.h
/opt/kernelgen/include/llvm/Support/PrettyStackTrace.h
/opt/kernelgen/include/llvm/Support/Process.h
/opt/kernelgen/include/llvm/Support/Program.h
/opt/kernelgen/include/llvm/Support/RWMutex.h
/opt/kernelgen/include/llvm/Support/Recycler.h
/opt/kernelgen/include/llvm/Support/RecyclingAllocator.h
/opt/kernelgen/include/llvm/Support/Regex.h
/opt/kernelgen/include/llvm/Support/Registry.h
/opt/kernelgen/include/llvm/Support/RegistryParser.h
/opt/kernelgen/include/llvm/Support/SMLoc.h
/opt/kernelgen/include/llvm/Support/Signals.h
/opt/kernelgen/include/llvm/Support/Solaris.h
/opt/kernelgen/include/llvm/Support/SourceMgr.h
/opt/kernelgen/include/llvm/Support/StreamableMemoryObject.h
/opt/kernelgen/include/llvm/Support/StringPool.h
/opt/kernelgen/include/llvm/Support/SwapByteOrder.h
/opt/kernelgen/include/llvm/Support/SystemUtils.h
/opt/kernelgen/include/llvm/Support/TargetFolder.h
/opt/kernelgen/include/llvm/Support/TargetRegistry.h
/opt/kernelgen/include/llvm/Support/TargetSelect.h
/opt/kernelgen/include/llvm/Support/ThreadLocal.h
/opt/kernelgen/include/llvm/Support/Threading.h
/opt/kernelgen/include/llvm/Support/TimeValue.h
/opt/kernelgen/include/llvm/Support/Timer.h
/opt/kernelgen/include/llvm/Support/ToolOutputFile.h
/opt/kernelgen/include/llvm/Support/TypeBuilder.h
/opt/kernelgen/include/llvm/Support/Valgrind.h
/opt/kernelgen/include/llvm/Support/ValueHandle.h
/opt/kernelgen/include/llvm/Support/Win64EH.h
/opt/kernelgen/include/llvm/Support/circular_raw_ostream.h
/opt/kernelgen/include/llvm/Support/raw_os_ostream.h
/opt/kernelgen/include/llvm/Support/raw_ostream.h
/opt/kernelgen/include/llvm/Support/system_error.h
/opt/kernelgen/include/llvm/Support/type_traits.h
/opt/kernelgen/include/llvm/SymbolTableListTraits.h
/opt/kernelgen/include/llvm/TableGen/Error.h
/opt/kernelgen/include/llvm/TableGen/Main.h
/opt/kernelgen/include/llvm/TableGen/Record.h
/opt/kernelgen/include/llvm/TableGen/TableGenAction.h
/opt/kernelgen/include/llvm/TableGen/TableGenBackend.h
/opt/kernelgen/include/llvm/Target/Mangler.h
/opt/kernelgen/include/llvm/Target/Target.td
/opt/kernelgen/include/llvm/Target/TargetCallingConv.h
/opt/kernelgen/include/llvm/Target/TargetCallingConv.td
/opt/kernelgen/include/llvm/Target/TargetData.h
/opt/kernelgen/include/llvm/Target/TargetELFWriterInfo.h
/opt/kernelgen/include/llvm/Target/TargetFrameLowering.h
/opt/kernelgen/include/llvm/Target/TargetInstrInfo.h
/opt/kernelgen/include/llvm/Target/TargetIntrinsicInfo.h
/opt/kernelgen/include/llvm/Target/TargetJITInfo.h
/opt/kernelgen/include/llvm/Target/TargetLibraryInfo.h
/opt/kernelgen/include/llvm/Target/TargetLowering.h
/opt/kernelgen/include/llvm/Target/TargetLoweringObjectFile.h
/opt/kernelgen/include/llvm/Target/TargetMachine.h
/opt/kernelgen/include/llvm/Target/TargetOpcodes.h
/opt/kernelgen/include/llvm/Target/TargetOptions.h
/opt/kernelgen/include/llvm/Target/TargetRegisterInfo.h
/opt/kernelgen/include/llvm/Target/TargetSchedule.td
/opt/kernelgen/include/llvm/Target/TargetSelectionDAG.td
/opt/kernelgen/include/llvm/Target/TargetSelectionDAGInfo.h
/opt/kernelgen/include/llvm/Target/TargetSubtargetInfo.h
/opt/kernelgen/include/llvm/Transforms/IPO.h
/opt/kernelgen/include/llvm/Transforms/IPO/InlinerPass.h
/opt/kernelgen/include/llvm/Transforms/IPO/PassManagerBuilder.h
/opt/kernelgen/include/llvm/Transforms/Instrumentation.h
/opt/kernelgen/include/llvm/Transforms/Scalar.h
/opt/kernelgen/include/llvm/Transforms/Utils/AddrModeMatcher.h
/opt/kernelgen/include/llvm/Transforms/Utils/BasicBlockUtils.h
/opt/kernelgen/include/llvm/Transforms/Utils/BasicInliner.h
/opt/kernelgen/include/llvm/Transforms/Utils/BuildLibCalls.h
/opt/kernelgen/include/llvm/Transforms/Utils/Cloning.h
/opt/kernelgen/include/llvm/Transforms/Utils/CmpInstAnalysis.h
/opt/kernelgen/include/llvm/Transforms/Utils/FunctionUtils.h
/opt/kernelgen/include/llvm/Transforms/Utils/Local.h
/opt/kernelgen/include/llvm/Transforms/Utils/ModuleUtils.h
/opt/kernelgen/include/llvm/Transforms/Utils/PromoteMemToReg.h
/opt/kernelgen/include/llvm/Transforms/Utils/SSAUpdater.h
/opt/kernelgen/include/llvm/Transforms/Utils/SSAUpdaterImpl.h
/opt/kernelgen/include/llvm/Transforms/Utils/SimplifyIndVar.h
/opt/kernelgen/include/llvm/Transforms/Utils/UnifyFunctionExitNodes.h
/opt/kernelgen/include/llvm/Transforms/Utils/UnrollLoop.h
/opt/kernelgen/include/llvm/Transforms/Utils/ValueMapper.h
/opt/kernelgen/include/llvm/Transforms/Vectorize.h
/opt/kernelgen/include/llvm/Type.h
/opt/kernelgen/include/llvm/Use.h
/opt/kernelgen/include/llvm/User.h
/opt/kernelgen/include/llvm/Value.h
/opt/kernelgen/include/llvm/ValueSymbolTable.h
/opt/kernelgen/include/polly/Cloog.h
/opt/kernelgen/include/polly/CodeGeneration.h
/opt/kernelgen/include/polly/Config/config.h
/opt/kernelgen/include/polly/Dependences.h
/opt/kernelgen/include/polly/LinkAllPasses.h
/opt/kernelgen/include/polly/MayAliasSet.h
/opt/kernelgen/include/polly/RegisterPasses.h
/opt/kernelgen/include/polly/ScheduleOptimizer.h
/opt/kernelgen/include/polly/ScopDetection.h
/opt/kernelgen/include/polly/ScopInfo.h
/opt/kernelgen/include/polly/ScopLib.h
/opt/kernelgen/include/polly/ScopPass.h
/opt/kernelgen/include/polly/Support/GICHelper.h
/opt/kernelgen/include/polly/Support/SCEVValidator.h
/opt/kernelgen/include/polly/Support/ScopHelper.h
/opt/kernelgen/include/polly/TempScopInfo.h
/opt/kernelgen/lib/BugpointPasses.so
/opt/kernelgen/lib/LLVMHello.so
/opt/kernelgen/lib/LLVMPolly.so
/opt/kernelgen/lib/libLLVM-3.1svn.so
/opt/kernelgen/lib/libLLVMAnalysis.a
/opt/kernelgen/lib/libLLVMArchive.a
/opt/kernelgen/lib/libLLVMAsmParser.a
/opt/kernelgen/lib/libLLVMAsmPrinter.a
/opt/kernelgen/lib/libLLVMBitReader.a
/opt/kernelgen/lib/libLLVMBitWriter.a
/opt/kernelgen/lib/libLLVMCBackendCodeGen.a
/opt/kernelgen/lib/libLLVMCBackendInfo.a
/opt/kernelgen/lib/libLLVMCodeGen.a
/opt/kernelgen/lib/libLLVMCore.a
/opt/kernelgen/lib/libLLVMDebugInfo.a
/opt/kernelgen/lib/libLLVMExecutionEngine.a
/opt/kernelgen/lib/libLLVMInstCombine.a
/opt/kernelgen/lib/libLLVMInstrumentation.a
/opt/kernelgen/lib/libLLVMInterpreter.a
/opt/kernelgen/lib/libLLVMJIT.a
/opt/kernelgen/lib/libLLVMLinker.a
/opt/kernelgen/lib/libLLVMMC.a
/opt/kernelgen/lib/libLLVMMCDisassembler.a
/opt/kernelgen/lib/libLLVMMCJIT.a
/opt/kernelgen/lib/libLLVMMCParser.a
/opt/kernelgen/lib/libLLVMObject.a
/opt/kernelgen/lib/libLLVMPTXAsmPrinter.a
/opt/kernelgen/lib/libLLVMPTXCodeGen.a
/opt/kernelgen/lib/libLLVMPTXDesc.a
/opt/kernelgen/lib/libLLVMPTXInfo.a
/opt/kernelgen/lib/libLLVMRuntimeDyld.a
/opt/kernelgen/lib/libLLVMScalarOpts.a
/opt/kernelgen/lib/libLLVMSelectionDAG.a
/opt/kernelgen/lib/libLLVMSupport.a
/opt/kernelgen/lib/libLLVMTableGen.a
/opt/kernelgen/lib/libLLVMTarget.a
/opt/kernelgen/lib/libLLVMTransformUtils.a
/opt/kernelgen/lib/libLLVMVectorize.a
/opt/kernelgen/lib/libLLVMX86AsmParser.a
/opt/kernelgen/lib/libLLVMX86AsmPrinter.a
/opt/kernelgen/lib/libLLVMX86CodeGen.a
/opt/kernelgen/lib/libLLVMX86Desc.a
/opt/kernelgen/lib/libLLVMX86Disassembler.a
/opt/kernelgen/lib/libLLVMX86Info.a
/opt/kernelgen/lib/libLLVMX86Utils.a
/opt/kernelgen/lib/libLLVMipa.a
/opt/kernelgen/lib/libLLVMipo.a
/opt/kernelgen/lib/libLTO.a
/opt/kernelgen/lib/libLTO.so
/opt/kernelgen/lib/libpollyanalysis.a
/opt/kernelgen/lib/libpollyexchange.a
/opt/kernelgen/lib/libpollyjson.a
/opt/kernelgen/lib/libpollysupport.a
/opt/kernelgen/lib/libprofile_rt.a
/opt/kernelgen/lib/libprofile_rt.so
/opt/kernelgen/share/man/man1/bugpoint.1
/opt/kernelgen/share/man/man1/lit.1
/opt/kernelgen/share/man/man1/llc.1
/opt/kernelgen/share/man/man1/lli.1
/opt/kernelgen/share/man/man1/llvm-ar.1
/opt/kernelgen/share/man/man1/llvm-as.1
/opt/kernelgen/share/man/man1/llvm-bcanalyzer.1
/opt/kernelgen/share/man/man1/llvm-config.1
/opt/kernelgen/share/man/man1/llvm-cov.1
/opt/kernelgen/share/man/man1/llvm-diff.1
/opt/kernelgen/share/man/man1/llvm-dis.1
/opt/kernelgen/share/man/man1/llvm-extract.1
/opt/kernelgen/share/man/man1/llvm-ld.1
/opt/kernelgen/share/man/man1/llvm-link.1
/opt/kernelgen/share/man/man1/llvm-nm.1
/opt/kernelgen/share/man/man1/llvm-prof.1
/opt/kernelgen/share/man/man1/llvm-ranlib.1
/opt/kernelgen/share/man/man1/opt.1
/opt/kernelgen/share/man/man1/tblgen.1
#
# GCC files.
#
/opt/kernelgen/bin/kernelgen-cpp
/opt/kernelgen/bin/kernelgen-gcc
/opt/kernelgen/bin/kernelgen-gcov
/opt/kernelgen/bin/kernelgen-gfortran
/opt/kernelgen/bin/x86_64-unknown-linux-gnu-gcc-4.6.3
/opt/kernelgen/bin/x86_64-unknown-linux-gnu-kernelgen-gcc
/opt/kernelgen/bin/x86_64-unknown-linux-gnu-kernelgen-gfortran
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtbegin.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtbeginS.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtbeginT.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtend.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtendS.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtfastmath.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtprec32.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtprec64.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/crtprec80.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgcc.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgcc_eh.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgcov.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgfortranbegin.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/32/libgfortranbegin.la
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtbegin.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtbeginS.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtbeginT.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtend.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtendS.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtfastmath.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtprec32.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtprec64.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/crtprec80.o
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib.f90
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib.mod
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/finclude/omp_lib_kinds.mod
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/README
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/include-fixed/X11/Xw32defs.h
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
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/fixinc_list
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/gsyslimits.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/include/README
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/include/limits.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/macro_list
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/mkheaders.conf
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgcc.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgcc_eh.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgcov.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgfortranbegin.a
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/libgfortranbegin.la
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ada/gcc-interface/ada-tree.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/alias.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/all-tree.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ansidecl.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/auto-host.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/b-header-vars
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/basic-block.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/bitmap.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/builtins.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/bversion.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-common.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-family/c-common.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-objc.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-pragma.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/c-pretty-print.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cfghooks.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cfgloop.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cgraph.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cif-code.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/dbxelf.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/elfos.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/glibc-stdint.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/gnu-user.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/att.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/biarch64.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/i386-protos.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/i386.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/linux64.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/unix.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/i386/x86-64.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/linux-android.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/linux.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/config/vxworks-dummy.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/configargs.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/coretypes.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cp/cp-tree.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cppdefault.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/cpplib.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/debug.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/defaults.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/diagnostic-core.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/diagnostic.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/diagnostic.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/double-int.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/emit-rtl.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/except.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/filenames.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/fixed-value.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/flag-types.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/flags.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/function.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gcc-plugin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/genrtl.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ggc.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gimple.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gimple.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gsstruct.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/gtype-desc.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/hard-reg-set.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/hashtab.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/highlev-plugin-common.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/hwint.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/incpath.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/input.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-constants.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-flags.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-modes.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/insn-notes.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/intl.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-prop.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-ref-inline.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-ref.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-reference.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/ipa-utils.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/java/java-tree.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/langhooks.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/libiberty.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/line-map.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/machmode.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/md5.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/mode-classes.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/objc/objc-tree.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/obstack.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/omp-builtins.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/options.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/opts.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/output.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/params.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/params.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin-api.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin-version.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/plugin.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/pointer-set.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/predict.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/predict.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/prefix.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/pretty-print.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/real.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/reg-notes.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/rtl.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/rtl.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/safe-ctype.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/sbitmap.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/splay-tree.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/statistics.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/symtab.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/sync-builtins.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/system.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/target.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/target.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/timevar.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/timevar.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tm-preds.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tm.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tm_p.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/toplev.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-check.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-dump.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-flow-inline.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-flow.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-inline.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-iterator.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-pass.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-ssa-alias.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-ssa-operands.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree-ssa-sccvn.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/tree.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/treestruct.def
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/vec.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/vecir.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/vecprim.h
/opt/kernelgen/lib/gcc/x86_64-unknown-linux-gnu/4.6.3/plugin/include/version.h
/opt/kernelgen/%{lib32}/libgcc_s.so
/opt/kernelgen/%{lib32}/libgcc_s.so.1
/opt/kernelgen/%{lib32}/libgfortran.a
/opt/kernelgen/%{lib32}/libgfortran.la
/opt/kernelgen/%{lib32}/libgfortran.so
/opt/kernelgen/%{lib32}/libgfortran.so.3
/opt/kernelgen/%{lib32}/libgfortran.so.3.0.0
/opt/kernelgen/%{lib32}/libgfortran.spec
/opt/kernelgen/%{lib32}/libgomp.a
/opt/kernelgen/%{lib32}/libgomp.la
/opt/kernelgen/%{lib32}/libgomp.so
/opt/kernelgen/%{lib32}/libgomp.so.1
/opt/kernelgen/%{lib32}/libgomp.so.1.0.0
/opt/kernelgen/%{lib32}/libgomp.spec
/opt/kernelgen/%{lib32}/libmudflap.a
/opt/kernelgen/%{lib32}/libmudflap.la
/opt/kernelgen/%{lib32}/libmudflap.so
/opt/kernelgen/%{lib32}/libmudflap.so.0
/opt/kernelgen/%{lib32}/libmudflap.so.0.0.0
/opt/kernelgen/%{lib32}/libmudflapth.a
/opt/kernelgen/%{lib32}/libmudflapth.la
/opt/kernelgen/%{lib32}/libmudflapth.so
/opt/kernelgen/%{lib32}/libmudflapth.so.0
/opt/kernelgen/%{lib32}/libmudflapth.so.0.0.0
/opt/kernelgen/%{lib32}/libquadmath.a
/opt/kernelgen/%{lib32}/libquadmath.la
/opt/kernelgen/%{lib32}/libquadmath.so
/opt/kernelgen/%{lib32}/libquadmath.so.0
/opt/kernelgen/%{lib32}/libquadmath.so.0.0.0
/opt/kernelgen/%{lib32}/libssp.a
/opt/kernelgen/%{lib32}/libssp.la
/opt/kernelgen/%{lib32}/libssp.so
/opt/kernelgen/%{lib32}/libssp.so.0
/opt/kernelgen/%{lib32}/libssp.so.0.0.0
/opt/kernelgen/%{lib32}/libssp_nonshared.a
/opt/kernelgen/%{lib32}/libssp_nonshared.la
/opt/kernelgen/lib64/libgcc_s.so
/opt/kernelgen/lib64/libgcc_s.so.1
/opt/kernelgen/lib64/libgfortran.a
/opt/kernelgen/lib64/libgfortran.la
/opt/kernelgen/lib64/libgfortran.so
/opt/kernelgen/lib64/libgfortran.so.3
/opt/kernelgen/lib64/libgfortran.so.3.0.0
/opt/kernelgen/lib64/libgfortran.spec
/opt/kernelgen/lib64/libgomp.a
/opt/kernelgen/lib64/libgomp.la
/opt/kernelgen/lib64/libgomp.so
/opt/kernelgen/lib64/libgomp.so.1
/opt/kernelgen/lib64/libgomp.so.1.0.0
/opt/kernelgen/lib64/libgomp.spec
/opt/kernelgen/%{lib64}/libiberty.a
/opt/kernelgen/lib64/libmudflap.a
/opt/kernelgen/lib64/libmudflap.la
/opt/kernelgen/lib64/libmudflap.so
/opt/kernelgen/lib64/libmudflap.so.0
/opt/kernelgen/lib64/libmudflap.so.0.0.0
/opt/kernelgen/lib64/libmudflapth.a
/opt/kernelgen/lib64/libmudflapth.la
/opt/kernelgen/lib64/libmudflapth.so
/opt/kernelgen/lib64/libmudflapth.so.0
/opt/kernelgen/lib64/libmudflapth.so.0.0.0
/opt/kernelgen/lib64/libquadmath.a
/opt/kernelgen/lib64/libquadmath.la
/opt/kernelgen/lib64/libquadmath.so
/opt/kernelgen/lib64/libquadmath.so.0
/opt/kernelgen/lib64/libquadmath.so.0.0.0
/opt/kernelgen/lib64/libssp.a
/opt/kernelgen/lib64/libssp.la
/opt/kernelgen/lib64/libssp.so
/opt/kernelgen/lib64/libssp.so.0
/opt/kernelgen/lib64/libssp.so.0.0.0
/opt/kernelgen/lib64/libssp_nonshared.a
/opt/kernelgen/lib64/libssp_nonshared.la
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/cc1
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/collect2
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/f951
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/fixinc.sh
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/fixincl
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/mkheaders
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/install-tools/mkinstalldirs
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.la
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.so
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.so.0
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/liblto_plugin.so.0.0.0
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/lto-wrapper
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.3/lto1
/opt/kernelgen/share/locale/be/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/be/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/ca/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/da/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/da/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/de/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/de/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/el/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/el/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/es/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/es/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/fi/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/fi/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/fr/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/fr/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/id/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/id/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/ja/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/ja/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/nl/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/nl/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/ru/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/ru/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/sr/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/sv/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/sv/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/tr/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/tr/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/uk/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/vi/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/vi/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/zh_CN/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/zh_CN/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/locale/zh_TW/LC_MESSAGES/cpplib.mo
/opt/kernelgen/share/locale/zh_TW/LC_MESSAGES/gcc.mo
/opt/kernelgen/share/man/man1/kernelgen-cpp.1
/opt/kernelgen/share/man/man1/kernelgen-gcc.1
/opt/kernelgen/share/man/man1/kernelgen-gcov.1
/opt/kernelgen/share/man/man1/kernelgen-gfortran.1
/opt/kernelgen/share/man/man7/fsf-funding.7
/opt/kernelgen/share/man/man7/gfdl.7
/opt/kernelgen/share/man/man7/gpl.7


#
# Add paths for binaries and libraries into the system-wide configs.
#
%post
echo "export PATH=\$PATH:/opt/kernelgen/bin" >>/etc/profile.d/kernelgen.sh
echo "/opt/kernelgen/lib" >>/etc/ld.so.conf.d/kernelgen.conf
echo "/opt/kernelgen/lib64" >>/etc/ld.so.conf.d/kernelgen.conf


%changelog
* Tue Sep 13 2011 Dmitry Mikushin <maemarcus@gmail.com> 0.2
- started preparing 0.2 "accurate" release
* Sun Jul 10 2011 Dmitry Mikushin <dmikushin@nvidia.com> 0.1
- initial release

