---
layout: default
title: "POGS"
---

POGS
====

POGS is a solver for convex optimization of the form

\\[
	\\begin{aligned}
    &\\text{minimize}
    & & f(y) + g(x)  \\\\\\
    & \\text{subject to} 
    & & y = A x,
	\\end{aligned}
\\]

where \\(f\\) and \\(g\\) are convex and can take on the values \\(\\mathbf{R} \\cup \\{\\infty\\}\\). This formulation is known as _graph form_. The solver is based on [ADMM](http://foges.github.io/pogs/ref/admm) meaning that it can solve _very large problems extremely quickly_. It is written in C++, supports both single and double precision arithmetic and comes with wrappers for MATLAB and R. Higher performance can be achieved by using the GPU (CUDA) version.


### Problem Classes

POGS can be used in a wide range of settings from Machine Learning, to Signal Processing or Finance. Here are examples of problems where POGS may be useful

  + Lasso, Ridge Regression, Logistic Regression, Huber Fitting and Elastic Net Regulariation,
  + Total Variation Denoising, Optimal Control,
  + Linear Programs and Quadratic Programs.

In particular, POGS can very efficiently solve problems that involve different objective functions, but the same data matrix (eg. tracing out a regularization path).

### Problem Structure

POGS can solve any convex problem where the objective \\(f + g\\) is separable, meaning that it an be written as

\\[
  f(y) + g(x) = \\sum\_{i=1}^m f\_i(y\_i) + \\sum\_{j=1}^n g\_j(x\_j),
\\]

where \\(f\_i : \\mathbf{R} \\to \\mathbf{R}\\)  and \\(g\_j : \\mathbf{R} \\to \\mathbf{R}\\). Each term \\(f_i\\) (resp. \\(g_i\\)), is assumed to be of the form
\\[
  c \\, h(a \\, x - b) + d \\, x + e \\, x ^ 2,
\\]

where \\(a, b, d \\in \\mathbf{R}\\), \\(c, d \\in \\mathbf{R}\_+\\) and \\(h : \\mathbf{R} \\to \\mathbf{R} \\cup \\{ \\infty \\}\\). Currently \\(h\\) can be one of 12 functions or 4 constraints


\\[
  \\begin{aligned}
    &h(x) = |x|,       &h(x) &= x \\log(x),     &h(x) &= e^x,          &h(x) &= \\text{huber}(x),&\\\\\\
    &h(x) = x,         &h(x) &= \\log(1 + e^x), &h(x) &= \\max(0, -x), &h(x) &= \\max(0, x), &\\\\\\
    &h(x) = -\\log(x), &h(x) &= 1/x,            &h(x) &=(1/2)x^2,      &h(x) &= 0, &\\\\\\
    &h(x) = I(x = 0),  &h(x) &= I(0 \\leq x),   &h(x) &= I(x \\geq 0), &h(x) &= I(0 \\leq x \\leq 0), &
  \\end{aligned}
\\]

where \\( I(x \\in \\mathcal{C}) \\) is the indicator function of the convex set \\(\\mathcal{C}\\) (taking on the value 0 if \\(x \\in \\mathcal{C}\\) and \\(\\infty\\) otherwise). Additional functions can be added by modifying the proximal operator header file `<pogs>/src/prox_lib.h`.


### Languages, Frameworks and Wrappers
Three different implementations are available

  1. C++/BLAS/OpenMP: A CPU version can be found in the file `<pogs>/src/pogs.cpp`. POGS must be linked to a BLAS library (such as the Apple Accelerate Framework or ATLAS).
  2. C++/cuBLAS/CUDA: A GPU version is located in the file `<pogs>/src/pogs.cu`. To use the GPU version, the CUDA SDK must be installed, and the computer must have a CUDA-capable GPU.
  3. MATLAB: A MATLAB implementation along with examples can be found in the `<pogs>/matlab` directory. The code is heavily documented and primarily intended for pedagogical purposes.

Wrappers for MATLAB and R can be found in the directories `<pogs>/src/interface_matlab` and `<pogs>/src/interface_r` respectively. See [MATLAB Setup](http://foges.github.io/pogs/stp/matlab) or [R Setup](http://foges.github.io/pogs/stp/r) for instructions.

### References
1. [Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd][block_splitting]
2. [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein][admm_distr_stats]
3. [Proximal Algorithms -- N. Parikh and S. Boyd][prox_algs]

[block_splitting]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd"

[admm_distr_stats]: http://www.stanford.edu/~boyd/papers/block_splitting.html "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein"

[prox_algs]: http://www.stanford.edu/~boyd/papers/prox_algs.html "Proximal Algorithms -- N. Parikh and S. Boyd"


### Author
Chris Fougner (fougner@stanford.edu)

Acknowledgement: POGS is partially based on work by Neal Parikh.

