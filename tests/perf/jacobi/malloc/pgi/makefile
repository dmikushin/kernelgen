EXTRA = -Mcuda=keepgpu,keepbin,keepptx,ptxinfo
PGF = pgfortran -fast -Mnomain -Minfo=accel -ta=nvidia:4.1,time $(EXTRA)

all: jacobi

jacobi: jacobi.o main.o
	$(PGF) jacobi.o main.o -o $@

main.o: ../main.c
	pgcc -c $< -o $@

jacobi.o: ../jacobi.f90
	$(PGF) -c $< -o $@

clean:
	rm -rf *.o jacobi

