#include <cstdio>

#include "examples.h"

typedef double real_t;

int main() {
  double t;
  printf("\nLasso.\n");
  t = Lasso<real_t>(1000, 100, 10000);
  printf("Solver Time: %e sec\n", t);

  printf("\nLasso Path.\n");
  t = LassoPath<real_t>(200, 1000, 10000);
  printf("Solver Time: %e sec\n", t);

  return 0;
}

