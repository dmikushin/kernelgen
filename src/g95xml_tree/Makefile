CC = gcc -fPIC
CFLAGS_ARCH=
CFLAGS=$(CFLAGS_ARCH) -DG95X_CHECK -g -I. -Icpp -Imd5 -g -Wall -DG95X_64BITS 
FC=g95xml-tree

all: $(FC)

$(FC): arith.o array.o bbt.o bigint.o bigreal.o check.o decl.o error.o expr.o format.o interface.o intrinsic.o io.o iresolve.o kinds.o lang-options.o match.o matchexp.o misc.o module.o parse.o primary.o resolve.o scalarize.o scanner.o select.o simplify.o st.o symbol.o forall.o imodule.o entry.o xml-basic.o xml-print.o xml-cpp.o dump.o show.o xml-bbt.o xml-object.o xml-dump-body.o xml-dump-attr.o xml-locate.o xml-dump-source.o xml-get-locus.o xml-example.o cpp/libcpp.a md5/libmd5.a
	cd cpp ; $(MAKE)
	cd md5 ; $(MAKE)
	$(CC) -o $(FC) arith.o array.o bbt.o bigint.o bigreal.o check.o decl.o error.o expr.o format.o interface.o intrinsic.o io.o iresolve.o kinds.o lang-options.o match.o matchexp.o misc.o module.o parse.o primary.o resolve.o scalarize.o scanner.o select.o simplify.o st.o symbol.o forall.o imodule.o entry.o xml-basic.o xml-print.o xml-cpp.o xml-bbt.o xml-object.o xml-dump-body.o xml-dump-attr.o xml-locate.o xml-dump-source.o xml-get-locus.o xml-example.o dump.o show.o -Lcpp -lcpp -liberty -Lmd5 -lmd5

arith.o: arith.c
	$(CC) $(CFLAGS) -c arith.c
array.o: array.c
	$(CC) $(CFLAGS) -c array.c
bbt.o: bbt.c
	$(CC) $(CFLAGS) -c bbt.c
bigint.o: bigint.c
	$(CC) $(CFLAGS) -c bigint.c
bigreal.o: bigreal.c
	$(CC) $(CFLAGS) -c bigreal.c
check.o: check.c
	$(CC) $(CFLAGS) -c check.c
decl.o: decl.c xml-decl-funcs.c
	$(CC) $(CFLAGS) -c decl.c
error.o: error.c
	$(CC) $(CFLAGS) -c error.c
expr.o: expr.c
	$(CC) $(CFLAGS) -c expr.c
format.o: format.c
	$(CC) $(CFLAGS) -c format.c
interface.o: interface.c xml-interface-funcs.c
	$(CC) $(CFLAGS) -c interface.c
intrinsic.o: intrinsic.c xml-intrinsic-funcs.c
	$(CC) $(CFLAGS) -c intrinsic.c
io.o: io.c
	$(CC) $(CFLAGS) -c io.c
iresolve.o: iresolve.c xml-iresolve-funcs.c
	$(CC) $(CFLAGS) -c iresolve.c
kinds.o: kinds.c
	$(CC) $(CFLAGS) -c kinds.c
lang-options.o: lang-options.c
	$(CC) $(CFLAGS) -c lang-options.c
match.o: match.c xml-match-funcs.c
	$(CC) $(CFLAGS) -c match.c
matchexp.o: matchexp.c
	$(CC) $(CFLAGS) -c matchexp.c
misc.o: misc.c
	$(CC) $(CFLAGS) -c misc.c
module.o: module.c xml-module-funcs.c
	$(CC) $(CFLAGS) -c module.c
parse.o: parse.c xml-parse-funcs.c
	$(CC) $(CFLAGS) -c parse.c
primary.o: primary.c 
	$(CC) $(CFLAGS) -c primary.c
resolve.o: resolve.c
	$(CC) $(CFLAGS) -c resolve.c
scalarize.o: scalarize.c
	$(CC) $(CFLAGS) -c scalarize.c
scanner.o: scanner.c xml-scanner-funcs.c
	$(CC) $(CFLAGS) -c scanner.c
select.o: select.c
	$(CC) $(CFLAGS) -c select.c
simplify.o: simplify.c
	$(CC) $(CFLAGS) -c simplify.c
st.o: st.c
	$(CC) $(CFLAGS) -c st.c
symbol.o: symbol.c xml-symbol-funcs.c
	$(CC) $(CFLAGS) -c symbol.c
forall.o: forall.c
	$(CC) $(CFLAGS) -c forall.c
imodule.o: imodule.c
	$(CC) $(CFLAGS) -c imodule.c
entry.o: entry.c
	$(CC) $(CFLAGS) -c entry.c
xml-basic.o: xml-basic.c
	$(CC) $(CFLAGS) -c xml-basic.c
xml-print.o: xml-print.c xml-string.h
	$(CC) $(CFLAGS) -c xml-print.c
xml-cpp.o: xml-cpp.c
	$(CC) $(CFLAGS) -c xml-cpp.c
xml-bbt.o: xml-bbt.c
	$(CC) $(CFLAGS) -c xml-bbt.c
xml-locate.o: xml-locate.c
	$(CC) $(CFLAGS) -c xml-locate.c
xml-object.o: xml-object.c xml-string.h
	$(CC) $(CFLAGS) -c xml-object.c
xml-dump.o: xml-dump.c
	$(CC) $(CFLAGS) -c xml-dump.c
dump.o: dump.c
	$(CC) $(CFLAGS) -c dump.c
show.o: show.c
	$(CC) $(CFLAGS) -c show.c

xml-dump-attr.o: xml-dump-attr.h xml-dump-body.h xml-dump.c xml-dump.h xml-string.h
	$(CC) $(CFLAGS) -c -o xml-dump-attr.o -DG95X_DUMP_ATTR  xml-dump.c

xml-dump-body.o: xml-dump-attr.h xml-dump-body.h xml-dump.c xml-dump.h xml-string.h
	$(CC) $(CFLAGS) -c -o xml-dump-body.o -DG95X_DUMP_BODY  xml-dump.c

xml-dump-source.o: xml-dump-source.c xml-string.h
	$(CC) $(CFLAGS) -c -o xml-dump-source.o xml-dump-source.c

xml-example.o: xml-example.c
	$(CC) $(CFLAGS) -c -o xml-example.o xml-example.c

xml-get-locus.o: xml-get-locus.c 
	$(CC) $(CFLAGS) -c xml-get-locus.c

xml-dump-attr.h: xml-dump.c
	cpp -DG95X_DUMP_ATTR -DINTF xml-dump.c > xml-dump-attr.h

xml-dump-body.h: xml-dump.c
	cpp -DG95X_DUMP_BODY -DINTF xml-dump.c > xml-dump-body.h

clean:
	cd cpp && $(MAKE) clean
	cd md5 && $(MAKE) clean
	\rm -f arith.o array.o bbt.o bigint.o bigreal.o check.o decl.o error.o expr.o format.o interface.o intrinsic.o io.o iresolve.o kinds.o lang-options.o match.o matchexp.o misc.o module.o parse.o primary.o resolve.o scalarize.o scanner.o select.o simplify.o st.o symbol.o forall.o imodule.o entry.o xml-basic.o xml-print.o xml-cpp.o $(FC) *~ dump.o show.o xml-bbt.o xml-object.o xml-dump-body.o xml-dump-attr.o xml-dump-body.h xml-dump-attr.h xml-locate.o xml-dump-source.o xml-get-locus.o xml-example.o
	\rm -rf examples

cpp/libcpp.a: 
	cd cpp && $(MAKE)

md5/libmd5.a: 
	cd md5 && $(MAKE)





