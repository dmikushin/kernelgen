EXTRA = -Mcuda=keepgpu,keepbin,keepptx,ptxinfo
PGC = pgcc -fast -Minfo=accel -ta=nvidia:4.1,time $(EXTRA)

all: matmul

matmul: matmul.o main.o
	$(PGC) matmul.o main.o -o $@

main.o: ../main.c
	pgcc -c $< -o $@

matmul.o: ../matmul.c
	$(PGC) -c $< -o $@

clean:
	rm -rf *.o matmul

