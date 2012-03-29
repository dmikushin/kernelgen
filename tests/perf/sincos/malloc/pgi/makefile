EXTRA = -Mcuda=keepgpu,keepbin,keepptx,ptxinfo
PGF = pgfortran -fast -Mnomain -Minfo=accel -ta=nvidia:4.1,time $(EXTRA)

all: sincos

sincos: sincos.o main.o
	$(PGF) sincos.o main.o -o $@

main.o: ../main.c
	pgcc -c $< -o $@

sincos.o: ../sincos.f90
	$(PGF) -c $< -o $@

clean:
	rm -rf *.o sincos

