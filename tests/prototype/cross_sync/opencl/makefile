CCOMP = gcc -g -std=c99 -I/opt/cuda/include -I/opt/AMDAPP/include -DVERBOSE

OBJECTS = builder.o error.o cross_sync.o

all: cross_sync

cross_sync: $(OBJECTS)
	$(CCOMP) $(OBJECTS) -o $@ -L/opt/AMDAPP/lib/x86 -L/opt/AMDAPP/lib/x86_64 -lOpenCL

builder.o: builder.c
	$(CCOMP) -c $< -o $@

error.o: error.c
	$(CCOMP) -c $< -o $@

cross_sync.o: cross_sync.c
	$(CCOMP) -c $< -o $@

clean:
	rm -rf cross_sync *.o

