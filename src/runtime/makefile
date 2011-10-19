all: 32/libkernelgen.a 64/libkernelgen.a

32/libkernelgen.a: 32/entry.o
	ar rcs $@ 32/entry.o

64/libkernelgen.a: 64/entry.o
	ar rcs $@ 64/entry.o

32/entry.o: entry.cpp
	g++ -m32 -c $< -o $@

64/entry.o: entry.cpp
	g++ -c $< -o $@

clean:
	rm -rf 32/*.o 32/*.a 64/*.o 64/*.a

