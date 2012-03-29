all: matmul

matmul: matmul.o main.o
	kernelgen-simple matmul.o main.o -o $@

main.o: ../main.c
	kernelgen-simple -std=c99 -c $< -o $@

matmul.o: ../matmul.c
	kernelgen-simple -std=c99 -c $< -o $@

clean:
	rm -rf *.o matmul

