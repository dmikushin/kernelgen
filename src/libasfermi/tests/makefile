.PHONY: constant2 dyloader sum_kernel

all: constant2 dyloader sum_kernel
#all: dyloader

consant2:
	cd $@ && $(MAKE)

sum_kernel:
	cd $@ && $(MAKE)

dyloader:
	cd $@ && $(MAKE)

clean:
	cd constant2 && $(MAKE) clean
	cd sum_kernel && $(MAKE) clean
	cd dyloader && $(MAKE) clean

