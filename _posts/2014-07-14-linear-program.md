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
A = matrix(rnorm(100 * 10), 100, 10)
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

~~~

This example can be found in the file `<pogs>/examples/cpp/lp_eq.m`.

