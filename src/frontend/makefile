all: kernelgen

kernelgen: kernelgen.o compile.o link.o
	g++ kernelgen.o compile.o link.o -o $@ \
		-L/opt/kernelgen/lib -lLLVM-3.0svn -L../util/64 -lutil -lelf

kernelgen.o: kernelgen.cpp
	g++ -g -I.. -c $< -o $@

compile.o: compile.cpp
	g++ -g -I.. -I/home/marcusmae/rpmbuild/BUILD/llvm/build/include \
		-D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -c $< -o $@

link.o: link.cpp
	g++ -g -I.. -I/home/marcusmae/rpmbuild/BUILD/llvm/build/include \
		-D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -c $< -o $@

clean:
	rm -rf *.o kernelgen

