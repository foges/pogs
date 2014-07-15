---
layout: page
title: "Huber Fitting"
category: egs
date: 2014-07-14 17:37:51
---

Consider the Robust Regression problem

\\[
  \\text{minimize} ~\\sum\_{i=1}^m\\text{huber}(a\_i^T x - b\_i),
\\]

where

\\[
  \\text{huber}(x) = \\left\\{\begin{aligned} &(1/2)x^2 & \|x\| \\leq 1 \\\\\\ &\|x\| - (1/2) & \|x\| > 1 \\end{aligned} \\right.
\\]

which has the graph form representation

\\[
	\\begin{aligned}
    &\\text{minimize}
    & & \\sum\_{i=1}^m\\text{huber}(y_i - b\_i) \\\\\\
    & \\text{subject to}
    & & y = A x.
	\\end{aligned}
\\]

or equivalently

\\[
	\\begin{aligned}
    &\\text{minimize}
    & & f(y) + g(x)  \\\\\\
    & \\text{subject to}
    & & y = A x,
	\\end{aligned}
\\]

where

\\[
  f_i(y\_i) = \\text{huber}(y_i - b\_i), ~~\\text{ and } ~~g\_j(x\_j) = 0.
\\]


### MATLAB Code

~~~ matlab
% Generate Data
A = randn(100, 10);
b = randn(100, 1);

% Populate f and g
f.h = kHuber;
f.b = b;
g.h = kZero;

% Solve
x = pogs(A, f, g);
~~~

This example can be found in the file `<pogs>/examples/matlab/huber_fit.m`.


### R Code

~~~ r
# Generate Data
A = matrix(rnorm(100 * 10), 100, 10)
b = rnorm(100)

# Populate f and g
f = list(h = kHuber(), b = b)
g = list(h = kZero())

# Solve
solution = pogs(A, f, g)
~~~

This example can be found in the file `<pogs>/examples/r/huber_fit.R`.


### C++ Code

~~~ c
#include <random>
#include <vector>

#include "pogs.h"

int main() {
  // Generate Data
  size_t m = 100, n = 10;
  std::vector<double> A(m * n);
  std::vector<double> b(m);
  std::vector<double> x(n);
  std::vector<double> y(m);

  std::default_random_engine generator;
  std::normal_distribution<double> n_dist(0.0, 1.0);

  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = n_dist(generator);

  for (unsigned int j = 0; j < n; ++j)
    b[i] = n_dist(generator);

  // Populate f and g
  PogsData<double, double*> pogs_data(A.data(), m, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();

  pogs_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(kHuber, 1.0, b[i]);

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kZero);

  // Solve
  Pogs(&pogs_data);
}
~~~

This example can be found in the file `<pogs>/examples/cpp/huber_fit.cpp`.

