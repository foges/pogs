---
layout: page
title: "Linear Program"
category: egs
date: 2014-07-14 17:37:24
---

Consider the equality-form Linear Program

\\[
	\\begin{aligned}
    &\\text{minimize}
    & & c^T x \\\\\\
    & \\text{subject to}
    & & A x = b \\\\\\
    & & & x \\geq 0,
	\\end{aligned}
\\]

which has the graph form representation

\\[
	\\begin{aligned}
    &\\text{minimize}
    & & I(y\_{1..m} = b) + y\_{m+1} + I(x \\geq 0)  \\\\\\
    & \\text{subject to}
    & & y = \\begin{bmatrix}A \\\\\\ c^T \\end{bmatrix} x
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
  f_i(y\_i) = \\left\\{\\begin{aligned} & I(y\_i = b\_i) & i = 1\\ldots m \\\\ & y\_i & i = m+1 \\end{aligned} \\right., ~~\\text{ and } ~~g\_j(x\_j) = I(x\_j \\geq 0).
\\]

### MATLAB Code

~~~ matlab
% Generate Data
A = rand(100, 10);
b = A * rand(10, 1);
c = rand(10, 1);

% Populate f and g
f.h = [kIndEq0(100); kIdentity];
f.b = [b; 0];
g.h = kIndGe0;

% Solve.
x = pogs([A; c'], f, g);
~~~

This example can be found in the file `<pogs>/examples/matlab/lp_eq.m`.


### R Code

~~~ r
# Generate Data
A = matrix(runif(100 * 10), 100, 10)
b = A %*% runif(10)
c = runif(10);

# Populate f and g
f = list(h = c(kIndEq0(100), kIdentity()), b = c(b, 0))
g = list(h = kIndGe0())

# Solve.
solution = pogs(rbind(A, c), f, g)
~~~

This example can be found in the file `<pogs>/examples/r/lp_eq.m`.


### C++ Code

~~~ cpp
#include <random>
#include <vector>

#include "pogs.h"

int main() {
  // Generate Data
  size_t m = 100, n = 10;
  std::vector<double> A((m + 1) * n);
  std::vector<double> b(m);
  std::vector<double> x(n);
  std::vector<double> y(m + 1);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> u_dist(0.0, 1.0);

  for (unsigned int i = 0; i < (m + 1) * n; ++i)
    A[i] = u_dist(generator);

  for (unsigned int i = 0; i < n; ++i)
    v[i] = u_dist(generator);

  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      b[i] += A[i * n + j] * v[j];

  // Populate f and g
  PogsData<double, double*> pogs_data(A.data(), m + 1, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();

  pogs_data.f.reserve(m + 1);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(kIndEq0, 1.0, b[i]);
  pogs_data.f.emplace_back(kIdentity);

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kIndGe0);

  // Solve
  Pogs(&pogs_data);
}
~~~

This example can be found in the file `<pogs>/examples/cpp/lp_eq.m`.

