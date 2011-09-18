all: kernelgen

kernelgen: kernelgen.o execute.o
	g++ kernelgen.o execute.o -o $@ -L/opt/kernelgen/lib -lLLVM-3.0svn -L../util/64 -lutil -lelf

kernelgen.o: kernelgen.cpp
	g++ -g -I../util -I/home/marcusmae/rpmbuild/BUILD/llvm/build/include -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -c $< -o $@

execute.o: execute.cpp
	g++ -g -c $< -o $@

clean:
	rm -rf *.o kernelgen

