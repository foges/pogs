# To build with openmp set IFLAGS=-fopenmp

# Bulid directory
OBJDIR=build


# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -O3 -Wall -std=c++11 -fPIC #-DDEBUG # -Wconversion

# CUDA Flags
CUXX=nvcc
CUFLAGS=$(IFLAGS) -arch=sm_20 -Xcompiler -fPIC #-DDEBUG


# POGS header files.
POGS_HDR=\
	include/interface_defs.h \
	include/pogs.h \
	include/prox_lib.h \
	include/util.h \
	include/matrix/matrix.h \
	include/matrix/matrix_dense.h \
	include/matrix/matrix_sparse.h \
	include/projector/projector_cgls.h \
	include/projector/projector_direct.h

# CPU Specific headers and object files.
GSL_HDR=\
	cpu/include/gsl/cblas.h \
	cpu/include/gsl/gsl_blas.h \
	cpu/include/gsl/gsl_linalg.h \
	cpu/include/gsl/gsl_matrix.h \
	cpu/include/gsl/gsl_rand.h \
	cpu/include/gsl/gsl_spblas.h \
	cpu/include/gsl/gsl_spmat.h \
	cpu/include/gsl/gsl_vector.h

CPU_HDR=\
	cpu/include/cgls.h \
	cpu/include/equil_helper.h \
	cpu/include/projector_helper.h
CPU_MTX_OBJ=\
	$(OBJDIR)/cpu/matrix/matrix_sparse.o \
	$(OBJDIR)/cpu/matrix/matrix_dense.o
CPU_PRJ_OBJ=\
	$(OBJDIR)/cpu/projector/projector_cgls.o \
	$(OBJDIR)/cpu/projector/projector_direct_dense.o
CPU_OBJ=$(OBJDIR)/cpu/pogs.o

# GPU Specific headers and object files.
CML_HDR=\
	gpu/include/cml/cblas.h \
	gpu/include/cml/cml_blas.cuh \
	gpu/include/cml/cml_defs.cuh \
	gpu/include/cml/cml_linalg.cuh \
	gpu/include/cml/cml_matrix.cuh \
	gpu/include/cml/cml_rand.cuh \
	gpu/include/cml/cml_spblas.cuh \
	gpu/include/cml/cml_spmat.cuh \
	gpu/include/cml/cml_utils.cuh \
	gpu/include/cml/cml_vector.cuh

GPU_HDR=\
	gpu/include/cgls.cuh \
	gpu/include/equil_helper.cuh \
	gpu/include/projector_helper.cuh
GPU_MTX_OBJ=\
	$(OBJDIR)/gpu/matrix/matrix_dense.o \
	$(OBJDIR)/gpu/matrix/matrix_sparse.o
GPU_PRJ_OBJ=\
	$(OBJDIR)/gpu/projector/projector_cgls.o \
	$(OBJDIR)/gpu/projector/projector_direct_dense.o
GPU_OBJ=$(OBJDIR)/gpu/pogs.o


# Set vpath for build
VPATH=cpu cpu/matrix cpu/projector gpu gpu/matrix gpu/projector


# Build all
cpu: $(CPU_OBJ) $(CPU_MTX_OBJ) $(CPU_PRJ_OBJ)
	ar cr $(OBJDIR)/pogs.a $^

gpu: $(OBJDIR)/pogs_link.o $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ)
	ar cr $(OBJDIR)/pogs.a $^


# Directories
$(OBJDIR):
	mkdir -p $@

# Directories CPU
$(OBJDIR)/cpu: $(OBJDIR)
	mkdir -p $@ 

$(OBJDIR)/cpu/matrix: $(OBJDIR)/cpu
	mkdir -p $@ 

$(OBJDIR)/cpu/projector: $(OBJDIR)/cpu
	mkdir -p $@ 

# Directories GPU
$(OBJDIR)/gpu: $(OBJDIR)
	mkdir -p $@ 

$(OBJDIR)/gpu/matrix: $(OBJDIR)/gpu
	mkdir -p $@ 

$(OBJDIR)/gpu/projector: $(OBJDIR)/gpu
	mkdir -p $@ 


# POGS CPU objects
$(OBJDIR)/cpu/pogs.o: cpu/pogs.cpp $(POGS_HDR) $(GSL_HDR) | $(OBJDIR)/cpu
	$(CXX) -I include -Icpu/include $< $(CXXFLAGS) -c -o $@ 

$(OBJDIR)/cpu/matrix/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/matrix
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/projector/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu/projector
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

$(OBJDIR)/cpu/%.o: %.cpp $(CPU_HDR) | $(OBJDIR)/cpu
	$(CXX) -Iinclude -Icpu/include $< $(CXXFLAGS) $(IFLAGS) -c -o $@

# POGS GPU objects
$(OBJDIR)/pogs_link.o: $(GPU_OBJ) $(GPU_MTX_OBJ) $(GPU_PRJ_OBJ) | $(OBJDIR)
	$(CUXX) $(CUFLAGS) $^ -dlink -o $@ 

$(OBJDIR)/gpu/pogs.o: gpu/pogs.cu $(POGS_HDR) $(CML_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) -dc -o $@ 

$(OBJDIR)/gpu/matrix/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/matrix
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/projector/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu/projector
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

$(OBJDIR)/gpu/%.o: %.cu $(GPU_HDR) | $(OBJDIR)/gpu
	$(CUXX) -Iinclude -Igpu/include $< $(CUFLAGS) $(IFLAGS) -dc -o $@

clean:
	rm -rf pogs.a build/ *.dSYM

