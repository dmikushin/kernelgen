APPS = main1 main2 main3 main4 main5 main6

all: $(APPS)

LEVEL = ../..

include ../../makefile.in

main1: main1.o
	$(VERBOSE)kernelgen-gcc $< -o $@
	$(VERBOSE)./main1 > Output/main1.out-ref

main1.o: main1.c
	$(VERBOSE)kernelgen-gcc -c $< -o $@

main2: main2.o
	$(VERBOSE)kernelgen-gcc $< -o $@
	$(VERBOSE)./main2 > Output/main2.out-ref

main2.o: main2.c
	$(VERBOSE)kernelgen-gcc -c $< -o $@

main3: main3.o
	$(VERBOSE)kernelgen-gcc $< -o $@
	$(VERBOSE)./main3 > Output/main3.out-ref

main3.o: main3.c
	$(VERBOSE)kernelgen-gcc -c $< -o $@

main4: main4.o
	$(VERBOSE)kernelgen-gcc $< -o $@
	$(VERBOSE)./main4 > Output/main4.out-ref

main4.o: main4.c
	$(VERBOSE)kernelgen-gcc -c $< -o $@

main5: main5.o
	$(VERBOSE)kernelgen-gcc $< -o $@
	$(VERBOSE)./main5 > Output/main5.out-ref

main5.o: main5.c
	$(VERBOSE)kernelgen-gcc -c $< -o $@

main6: main6.o
	$(VERBOSE)kernelgen-gcc $< -o $@
	$(VERBOSE)./main6 > Output/main6.out-ref

main6.o: main6.c
	$(VERBOSE)kernelgen-gcc -c $< -o $@
