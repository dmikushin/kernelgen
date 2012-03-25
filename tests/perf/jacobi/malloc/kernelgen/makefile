all: jacobi

jacobi: jacobi.o main.o
	kernelgen-simple jacobi.o main.o -o $@

main.o: ../main.c
	kernelgen-simple -std=c99 -c $< -o $@

jacobi.o: ../jacobi.f90
	kernelgen-simple -c $< -o $@

clean:
	rm -rf *.o jacobi

