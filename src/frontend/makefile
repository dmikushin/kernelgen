all: kernelgen

kernelgen: kernelgen.cpp
	g++ -g -I../util -I/home/marcusmae/rpmbuild/BUILD/llvm/build/include \
		-D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS kernelgen.cpp \
		-o $@ -L/opt/kernelgen/lib -lLLVM-3.0svn -L../util/64 -lutil -lelf

clean:
	rm -rf kernelgen

