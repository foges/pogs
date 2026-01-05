#include <stdio.h>
#include <stdlib.h>
#include "../../src/interface_c/pogs_c.h"

// Test the C cone interface with a simple LP:
//   minimize    x1 + 0*x2
//   subject to  x1 + x2 = 2
//               x1, x2 >= 0
//
// In cone form:
//   minimize c^T * x where c = [1, 0]
//   subject to b - Ax ∈ K_y, x ∈ K_x
//   where A = [[1, 1]], b = [2]
//   K_y = {zero cone} (equality constraint)
//   K_x = {non-negative cone} (x >= 0)

int main() {
  // Problem dimensions
  size_t m = 1;  // Number of constraints
  size_t n = 2;  // Number of variables

  // Matrix A (row-major): [1, 1]
  double A[] = {1.0, 1.0};

  // Vectors b and c
  double b[] = {2.0};
  double c[] = {1.0, 0.0};

  // Define cone constraints for y: b - Ax = 0
  unsigned int y_indices[] = {0};
  struct ConeConstraintC cone_y = {CONE_ZERO, y_indices, 1};
  struct ConeConstraintC cones_y[] = {cone_y};

  // Define cone constraints for x: x >= 0
  unsigned int x_indices[] = {0, 1};
  struct ConeConstraintC cone_x = {CONE_NON_NEG, x_indices, 2};
  struct ConeConstraintC cones_x[] = {cone_x};

  // Allocate solution vectors
  double x[2], y[1], l[1];
  double optval;
  unsigned int final_iter;

  // Solver parameters
  double rho = 1.0;
  double abs_tol = 1e-6;
  double rel_tol = 1e-6;
  unsigned int max_iter = 10000;
  unsigned int verbose = 5;
  int adaptive_rho = 1;
  int gap_stop = 1;

  printf("Solving LP via C cone interface:\n");
  printf("  minimize    x1\n");
  printf("  subject to  x1 + x2 = 2\n");
  printf("              x1, x2 >= 0\n");
  printf("  Expected solution: x1 = 0, x2 = 2\n\n");

  // Solve
  int status = PogsConeD(ROW_MAJ, m, n, A, b, c, cones_x, 1, cones_y, 1,
                         rho, abs_tol, rel_tol, max_iter, verbose,
                         adaptive_rho, gap_stop, x, y, l, &optval, &final_iter);

  printf("\nSolution:\n");
  printf("  x1 = %.6f\n", x[0]);
  printf("  x2 = %.6f\n", x[1]);
  printf("  Optimal value = %.6f\n", optval);
  printf("  Status: %s\n", status == 0 ? "Success" : "Failed");
  printf("  Iterations: %u\n", final_iter);

  return status;
}
