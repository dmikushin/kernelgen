TARGETS = mini small standard large extralarge

.PHONY: $(TARGETS)

all: $(TARGETS)

mini:
	cd bin/mini && $(MAKE)
small:
	cd bin/small && $(MAKE)
standard:
	cd bin/standard && $(MAKE)
large:
	cd bin/large && $(MAKE)
extralarge:
	cd bin/extralarge && $(MAKE)

clean:
	cd bin/mini && $(MAKE) clean
	cd bin/small && $(MAKE) clean
	cd bin/standard && $(MAKE) clean
	cd bin/large && $(MAKE) clean
	cd bin/extralarge && $(MAKE) clean

