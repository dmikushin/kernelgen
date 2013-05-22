##===- makefile - KernelGen make rule for building Debian package ---------===//
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===//
##
## This build rule repacks Debian package from existing RPM build root.
##
##===----------------------------------------------------------------------===//

deb:
	rm -rf ~/rpmbuild/BUILDROOT/kernelgen
	mkdir -p ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN
	cp -rf control ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN/
	cp -rf postinst ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN
	chmod 0775 ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN/postinst
	cd ~/rpmbuild/BUILDROOT/kernelgen && \
		cp -rf ../opt . && \
		find . -depth -empty -type d -exec rmdir {} \; && \
		cd .. && dpkg-deb --build kernelgen kernelgen-0.2-accurate.x86_64.deb
