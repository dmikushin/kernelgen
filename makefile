.PHONY: src tests

all: src tests

src:
	cd src && $(MAKE)

install:
	cd src && $(MAKE) install

tests: src
	cd tests && $(MAKE) && $(MAKE) tests

clean:
	cd src && $(MAKE) clean
	cd tests && $(MAKE) clean

