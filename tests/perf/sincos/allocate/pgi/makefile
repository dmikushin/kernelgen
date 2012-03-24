all: sincos

sincos: sincos.o main.o
	pgfortran -Mnomain -Minfo=accel -ta=nvidia:4.1,time sincos.o main.o -o $@

main.o: ../main.c
	pgcc -c $< -o $@

sincos.o: ../sincos.f90
	pgfortran -fast -Minfo=accel -ta=nvidia:4.1,time -c $< -o $@

clean:
	rm -rf *.o sincos

