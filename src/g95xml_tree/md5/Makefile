CC = gcc -fPIC
CFLAGS=-Wall

all: libmd5.a

libmd5.a: md5.o
	ar rv libmd5.a md5.o

md5.o: md5.c md5.h
	$(CC) -c md5.c

clean:
	\rm -f *.o *.a

