all: privatize

privatize: privatize.o
	kernelgen-gcc $< -o $@

privatize.o: privatize.c
	kernelgen-gcc -c $< -o $@

clean:
	rm -rf privatize *.o

