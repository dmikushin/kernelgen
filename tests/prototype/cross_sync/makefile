CUPTIDIR = /opt/cuda/tools/CUPTI

all: cross_sync

cross_sync: cross_sync.cu
	nvcc -arch=sm_20 -I$(CUPTIDIR)/include $< -o $@ -L$(CUPTIDIR)/lib -lcupti -lpthread -lcuda

clean:
	rm -rf cross_sync

