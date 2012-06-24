TESTS = argcargv hello_c hello_f # privatize sincos

test0: $(TESTS)
	@cd argcargv && $(MAKE) test0
	@cd hello_c && $(MAKE) test0
	@cd hello_f && $(MAKE) test0

test1: $(TESTS)
	@cd argcargv && $(MAKE) test1
	@cd hello_c && $(MAKE) test1
	@cd hello_f && $(MAKE) test1

clean:
	@cd argcargv && $(MAKE) clean
	@cd hello_c && $(MAKE) clean
	@cd hello_f && $(MAKE) clean
	@cd privatize && $(MAKE) clean
	@cd sincos && $(MAKE) clean

