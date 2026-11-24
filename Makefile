NVCC = nvcc
CXXFLAGS = -O3 -lm
override NUMBER_OF_ROWS := 300
override NUMBER_OF_COLS := 300
override NUMBER_OF_MYOCYTE := 90000
PCG_ITERATIONS ?= 80
PCG_TOLERANCE ?= 1e-12
PCG_THREADS ?= 256
PCG_CHECK_INTERVAL ?= 4
NVCCFLAGS = -gencode arch=compute_35,code=sm_35 \
            -gencode arch=compute_37,code=sm_37 \
            -gencode arch=compute_61,code=sm_61 \
            -Dnumber_of_myocyte=$(NUMBER_OF_MYOCYTE) \
            -DMYO_ROWS=$(NUMBER_OF_ROWS) \
            -DMYO_COLS=$(NUMBER_OF_COLS) \
            -DPCG_MAX_ITER=$(PCG_ITERATIONS) \
            -DPCG_TOL=$(PCG_TOLERANCE) \
            -DPCG_THREADS_PER_BLOCK=$(PCG_THREADS) \
            -DPCG_CHECK_INTERVAL=$(PCG_CHECK_INTERVAL)
LIBS = -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -lcuda
EXECUTABLE = xNa_m$(NUMBER_OF_MYOCYTE).run
.PHONY: all clean
all: $(EXECUTABLE)
$(EXECUTABLE): main.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(LIBS) $< -o $@
clean:
	rm -f $(EXECUTABLE)
