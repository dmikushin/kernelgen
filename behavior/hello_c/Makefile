LEVEL = ../..
PROG     = hello_c
CPPFLAGS =
LDFLAGS  =

RUN_OPTIONS = -v1 $(PROJ_SRC_DIR)/test.scm

# Define USE_REFERENCE_OUTPUT which is a signal to Makefile.programs that the
# natively compiled program should not be run but that its output should be
# "faked" by copying the reference output. This is because gcc-4.6 miscompiles
# siod, resulting in a program that fails to parse the Fibonacci test.
USE_REFERENCE_OUTPUT := 1

include ../Makefile.multisrc

