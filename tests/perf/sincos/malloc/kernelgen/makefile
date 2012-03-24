all: sincos

sincos: sincos.o main.o
	kernelgen-simple sincos.o main.o -o $@

main.o: ../main.c
	kernelgen-simple -std=c99 -c $< -o $@

sincos.o: ../sincos.f90
	kernelgen-simple -c $< -o $@

clean:
	rm -rf *.o sincos

