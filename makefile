.PHONY: src tests

all: src tests

src:
	cd src && $(MAKE)

install:
	cd src && $(MAKE) install

tests: src
	cd tests && $(MAKE) && $(MAKE) tests

clean:
	cd src && $(MAKE) clean
	cd tests && $(MAKE) clean

deb:
	rm -rf ~/rpmbuild/BUILDROOT/kernelgen
	mkdir -p ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN
	cp -rf control ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN/
	cp -rf postinst ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN
	chmod 0775 ~/rpmbuild/BUILDROOT/kernelgen/DEBIAN/postinst
	cd ~/rpmbuild/BUILDROOT/kernelgen && \
		cp -rf ../kernelgen-0.2-accurate.x86_64/opt . && \
		find . -depth -empty -type d -exec rmdir {} \; && \
		cd .. && dpkg-deb --build kernelgen kernelgen-0.2-accurate.x86_64.deb
