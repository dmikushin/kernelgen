all: nesting

nesting: nesting.c
	kernelgen-gcc -std=c99 $< -o $@

clean:
	rm -rf nesting
