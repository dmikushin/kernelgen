LLVMDEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVMINC = -I/home/marcusmae/rpmbuild/BUILD/llvm/build/include
LLVMLIB = -L/opt/kernelgen/lib -lLLVM-3.0svn
UTILINC = -I..
UTILLIB = -L../util/64 -lutil -lelf

CXX = g++ -g

OBJECTS = kernelgen.o compile.o link.o

all: kernelgen

kernelgen: $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LLVMLIB) $(UTILLIB)

kernelgen.o: kernelgen.cpp
	$(CXX) $(UTILINC) -c $< -o $@

compile.o: compile.cpp
	$(CXX) $(UTILINC) $(LLVMINC) $(LLVMDEF) -c $< -o $@

link.o: link.cpp
	$(CXX) $(UTILINC) $(LLVMINC) $(LLVMDEF) -c $< -o $@

clean:
	rm -rf *.o kernelgen

