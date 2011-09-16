# Release name
%define release accurate

# Target operating system
%define target debian

# Build unoptimized version with debug info
%define debug 0

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
Source3:	ftp://upload.hpcforge.org/pub/kernelgen/kernelgen-r456.tar.gz
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
%endif
rm -rf $RPM_BUILD_DIR/kernelgen
tar -xf $RPM_SOURCE_DIR/kernelgen-r456.tar.gz


%if %fullrepack
%patch0 -p1
%patch1 -p1
%patch2 -p1
%patch3 -p1
%patch4 -p1
%patch5 -p1
%endif


%build
%if %fullrepack
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
../configure --enable-jit --enable-debug-runtime --enable-debug-symbols --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe --with-cloog=$RPM_BUILD_DIR/cloog --with-isl=$RPM_BUILD_DIR/cloog/isl --with-scoplib=$RPM_BUILD_DIR/scoplib-0.2.0
make -j%{njobs} CXXFLAGS=-O0
%else
../configure --enable-jit --enable-optimized --enable-shared --prefix=$RPM_BUILD_ROOT/opt/kernelgen --enable-targets=host,cbe --with-cloog=$RPM_BUILD_DIR/cloog --with-isl=$RPM_BUILD_DIR/cloog/isl --with-scoplib=$RPM_BUILD_DIR/scoplib-0.2.0
make -j%{njobs}
%endif
cd $RPM_BUILD_DIR/gcc-4.6
mkdir build
cd build/
../configure --prefix=$RPM_BUILD_ROOT/opt/kernelgen --program-prefix=kernelgen- --enable-languages=fortran --with-mpfr-include=/usr/include/ --with-mpfr-lib=/usr/lib64 --with-gmp-include=/usr/include/ --with-gmp-lib=/usr/lib64 --enable-plugin
%if %debug
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu make -j%{njobs} CFLAGS="-g -O0" CXXFLAGS="-g -O0"
%else
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu make -j%{njobs}
%endif
%endif
cd $RPM_BUILD_DIR/kernelgen/branches/accurate
./configure
make src


%install
rm -rf $RPM_BUILD_ROOT
mkdir -p $RPM_BUILD_ROOT/opt/kernelgen
cd $RPM_BUILD_ROOT/opt/kernelgen
%if (%target == fedora)
mkdir lib
mkdir lib64
ln -s lib lib32
%else
mkdir lib32
mkdir lib
ln -s lib lib64
%endif
cd $RPM_BUILD_DIR/cloog
make install
cd $RPM_BUILD_DIR/scoplib-0.2.0
make install
cd $RPM_BUILD_DIR/llvm/build
make install
cd $RPM_BUILD_DIR/gcc-4.6/build
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu make install
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libgcc_s.so
rm $RPM_BUILD_ROOT/opt/kernelgen/lib/libgcc_s.so.1
cd $RPM_BUILD_DIR/dragonegg
GCC=$RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcc LLVM_CONFIG=$RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config make clean
GCC=$RPM_BUILD_ROOT/opt/kernelgen/bin/kernelgen-gcc LLVM_CONFIG=$RPM_BUILD_ROOT/opt/kernelgen/bin/llvm-config make
cp dragonegg.so $RPM_BUILD_ROOT/opt/kernelgen/lib64/
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/lib/libLLVMPolly.a
rm -rf $RPM_BUILD_ROOT/opt/kernelgen/include/polly/CodeGeneration.h
cd $RPM_BUILD_DIR/kernelgen/branches/accurate
ROOT=$RPM_BUILD_ROOT make install

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
/opt/kernelgen/include64/kernelgen.kernelgen.mod
/opt/kernelgen/include64/kernelgen.h
/opt/kernelgen/include64/kernelgen.mod
/opt/kernelgen/include/kernelgen.kernelgen.mod
/opt/kernelgen/include/kernelgen.h
/opt/kernelgen/include/kernelgen.mod
/opt/kernelgen/lib64/dragonegg.so
/opt/kernelgen/lib64/libkernelgen.so
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/collect2
/opt/kernelgen/libexec/gcc/x86_64-unknown-linux-gnu/4.6.2/f951
/opt/kernelgen/lib/libcloog-isl.so.2.0.0
/opt/kernelgen/lib/libkernelgen.so
/opt/kernelgen/lib/libcloog-isl.so
/opt/kernelgen/lib/libcloog-isl.so.2
/opt/kernelgen/lib/libisl.so
/opt/kernelgen/lib/libisl.so.7
/opt/kernelgen/lib/libisl.so.7.0.0
/opt/kernelgen/lib/libLLVM-3.0svn.so
/opt/kernelgen/lib/libscoplib.so.0.0.0
/opt/kernelgen/lib/LLVMPolly.so

%post
echo "export PATH=\$PATH:/opt/kernelgen/bin" >>/etc/profile.d/kernelgen.sh
echo "/opt/kernelgen/lib" >>/etc/ld.so.conf.d/kernelgen.conf
echo "/opt/kernelgen/lib64" >>/etc/ld.so.conf.d/kernelgen.conf


%changelog
* Tue Sep 13 2011 Dmitry Mikushin <maemarcus@gmail.com> 0.2
- started preparing 0.2 "accurate" release
* Sun Jul 10 2011 Dmitry Mikushin <dmikushin@nvidia.com> 0.1
- initial release

