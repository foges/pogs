# To build with openmp set IFLAGS=-fopenmp

# User Vars
POGSROOT=../../src

# Example Files
EXSRC=lasso.cpp lasso_path.cpp

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -O3 -I$(POGSROOT)/include -std=c++11 -Wall

# CUDA Flags
CULDFLAGS_=-lcudart -lcublas -lcusparse
CUXXFLAGS=-D__CUDA

# Check System Args.
UNAME = $(shell uname -s)
ifeq ($(UNAME), Darwin)
LDFLAGS=-lm -framework Accelerate
CULDFLAGS=-L/usr/local/cuda/lib -L/usr/local/lib $(CULDFLAGS_)
else
LDFLAGS=-lm -lopenblas
CULDFLAGS=-L/usr/local/cuda/lib64 $(CULDFLAGS_)
endif

# CPU
cpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) cpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS) -I$(POGSROOT)/include -o run $(EXSRC) \
  $< $(POGSROOT)/build/pogs.a $(LDFLAGS)

# GPU
gpu: run_all.cpp examples.h $(EXSRC)
	$(MAKE) gpu -C $(POGSROOT) IFLAGS=$(IFLAGS)
	$(CXX) $(CXXFLAGS) $(CUXXFLAGS) -I$(POGSROOT)/include -o run $(EXSRC) \
  $< $(POGSROOT)/build/pogs.a $(CULDFLAGS)

clean:
	rm -f *.o *~ *~ run
	rm -rf *.dSYM

