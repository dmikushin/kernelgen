all: main1 main2 main3 main4 main5 main6

main1: main1.o
	kernelgen-gcc $< -o $@

main1.o: main1.c
	kernelgen-gcc -c $< -o $@

main2: main2.o
	kernelgen-gcc $< -o $@

main2.o: main2.c
	kernelgen-gcc -c $< -o $@

main3: main3.o
	kernelgen-gcc $< -o $@

main3.o: main3.c
	kernelgen-gcc -c $< -o $@

main4: main4.o
	kernelgen-gcc $< -o $@

main4.o: main4.c
	kernelgen-gcc -c $< -o $@

main5: main5.o
	kernelgen-gcc $< -o $@

main5.o: main5.c
	kernelgen-gcc -c $< -o $@

main6: main6.o
	kernelgen-gcc $< -o $@

main6.o: main6.c
	kernelgen-gcc -c $< -o $@

clean:
	rm -rf *.o main1 main2 main3 main4 main5 main6

