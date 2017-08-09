PREFIX=$(HOME)/torch/install
CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH -lpng
LDFLAGS_CPP=-L$(PREFIX)/lib -lluaT -lTH `pkg-config --libs opencv`

all: libcutils.so

libcutils.so: cutils.cu
	nvcc -arch sm_61 -O3 -DNDEBUG --compiler-options '-fPIC' -o libcutils.so --shared cutils.cu $(CFLAGS) $(LDFLAGS_NVCC)

clean:
		rm -f libcutils.so 
