PLUGIN_INC := $(dir $(shell gcc -print-libgcc-file-name))plugin/include

all: plugin.so

plugin.so: plugin.cpp
	g++ -g -I$(PLUGIN_INC) -fPIC -shared $< -o $@

test: test.cpp plugin.so
	g++ -g -fplugin=./plugin.so $< -o $@ && ./$@

clean:
	rm -rf plugin.so test

