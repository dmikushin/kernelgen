#!/bin/bash

make clean && make 

dim1=$1
dim2=$2
dim3=$3
size=$((800*800*800))
kernelgen_output=kernelgen_for.txt
normal_output=normal_for.txt

rm $kernelgen_output 
rm $normal_output
kernelgen_runmode=1 kernelgen_verbose=32 kernelgen_szheap=$size ./kernelgen_for $dim1 $dim2 $dim3 $kernelgen_output
./normal_for $dim1 $dim2 $dim3 $normal_output

echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
./test $kernelgen_output $normal_output
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
./test_float $kernelgen_output $normal_output
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

#kernelgen_runmode=1 kernelgen_szheap=8000 cuda-gdb ./kernelgen_for 10 10 10 kernelgen_for.txt
