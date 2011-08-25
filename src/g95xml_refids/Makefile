CFLAGS_ARCH=
CFLAGS=$(CFLAGS_ARCH) -DG95X_CHECK -g -I. -Icpp -g -Wall
CC=gcc -fPIC
FC=g95xml-refids
# arith.c array.c bbt.c bigint.c bigreal.c check.c decl.c error.c expr.c format.c interface.c intrinsic.c io.c iresolve.c kinds.c lang-options.c match.c matchexp.c misc.c module.c parse.c primary.c resolve.c scalarize.c scanner.c select.c simplify.c st.c symbol.c forall.c imodule.c entry.c xml-basic.c xml-traverse.c xml-print.c xml-cpp.c dump.c show.c
#

$(FC): arith.o array.o bbt.o bigint.o bigreal.o check.o decl.o error.o expr.o format.o interface.o intrinsic.o io.o iresolve.o kinds.o lang-options.o match.o matchexp.o misc.o module.o parse.o primary.o resolve.o scalarize.o scanner.o select.o simplify.o st.o symbol.o forall.o imodule.o entry.o xml-basic.o xml-traverse.o xml-print.o xml-cpp.o dump.o show.o xml-bbt.o cpp/libcpp.a
	$(CC) -o $(FC) arith.o array.o bbt.o bigint.o bigreal.o check.o decl.o error.o expr.o format.o interface.o intrinsic.o io.o iresolve.o kinds.o lang-options.o match.o matchexp.o misc.o module.o parse.o primary.o resolve.o scalarize.o scanner.o select.o simplify.o st.o symbol.o forall.o imodule.o entry.o xml-basic.o xml-traverse.o xml-print.o xml-cpp.o xml-bbt.o dump.o show.o -Lcpp -lcpp -liberty
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
interface.o: interface.c
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
parse.o: parse.c
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
symbol.o: symbol.c
	$(CC) $(CFLAGS) -c symbol.c
forall.o: forall.c
	$(CC) $(CFLAGS) -c forall.c
imodule.o: imodule.c
	$(CC) $(CFLAGS) -c imodule.c
entry.o: entry.c
	$(CC) $(CFLAGS) -c entry.c
xml-basic.o: xml-basic.c
	$(CC) $(CFLAGS) -c xml-basic.c
xml-traverse.o: xml-traverse.c
	$(CC) $(CFLAGS) -c xml-traverse.c
xml-print.o: xml-print.c
	$(CC) $(CFLAGS) -c xml-print.c
xml-cpp.o: xml-cpp.c
	$(CC) $(CFLAGS) -c xml-cpp.c
xml-bbt.o: xml-bbt.c
	$(CC) $(CFLAGS) -c xml-bbt.c
dump.o: dump.c
	$(CC) $(CFLAGS) -c dump.c
show.o: show.c
	$(CC) $(CFLAGS) -c show.c
clean:
	cd cpp ; $(MAKE) clean
	\rm -f arith.o array.o bbt.o bigint.o bigreal.o check.o decl.o error.o expr.o format.o interface.o intrinsic.o io.o iresolve.o kinds.o lang-options.o match.o matchexp.o misc.o module.o parse.o primary.o resolve.o scalarize.o scanner.o select.o simplify.o st.o symbol.o forall.o imodule.o entry.o xml-basic.o xml-traverse.o xml-print.o xml-cpp.o $(FC) *~ dump.o show.o xml-bbt.o


cpp/libcpp.a: 
	cd cpp && $(MAKE)





