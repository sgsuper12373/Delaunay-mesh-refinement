ARCH ?= sm_80
CCBIN ?=     # host compiler for nvcc, when the system g++ is newer than nvcc accepts
HEADERS = main.cu refine.cuh kernels.cuh mesh.cuh geometry.cuh

dmr: $(HEADERS)
	nvcc -O3 -std=c++17 -arch=$(ARCH) $(if $(CCBIN),-ccbin $(CCBIN)) -o dmr main.cu

clean:
	rm -f dmr

.PHONY: clean
