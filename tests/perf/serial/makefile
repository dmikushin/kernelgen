FLAGS=-g -O3 -std=c99

all: serial_cpu serial_gpu

serial_cpu: serial_cpu.o
	gcc $< -o $@

serial_cpu.o: serial.c
	gcc -g $(FLAGS) -c $< -o $@

serial_gpu: serial_gpu.o
	kernelgen-gcc $< -o $@

serial_gpu.o: serial.c
	kernelgen-gcc -g $(FLAGS) -c $< -o $@

clean:
	rm -rf serial_cpu serial_gpu *.o

