#include <iostream>
#include <vector>
#include <cmath>

#include "pogs.h"

// Simple test for SDP cone projection
// Test: Project a matrix with negative eigenvalues
int main() {
  // Test 1: 2x2 diagonal matrix with one negative eigenvalue
  // Matrix: [1 0; 0 -2]
  // Vectorized (column-major lower triangle): [1, 0, -2]
  // After projection: [1, 0, 0] (zero out negative eigenvalue)

  std::cout << "Test 1: 2x2 diagonal matrix [1, 0; 0, -2]" << std::endl;

  std::vector<double> v1 = {1.0, 0.0, -2.0};
  std::vector<CONE_IDX> idx1 = {0, 1, 2};

  ConeConstraintRaw cone1;
  cone1.cone = kConeSdp;
  cone1.idx = idx1.data();
  cone1.size = 3;

  ProxConeSdpCpu(cone1, v1.data());

  std::cout << "  Input:  [" << 1.0 << ", " << 0.0 << ", " << -2.0 << "]" << std::endl;
  std::cout << "  Output: [" << v1[0] << ", " << v1[1] << ", " << v1[2] << "]" << std::endl;
  std::cout << "  Expected: [1, 0, 0]" << std::endl;

  // Test 2: 2x2 matrix with both eigenvalues negative
  // Matrix: [-1 0; 0 -2]
  // After projection: [0, 0, 0]

  std::cout << "\nTest 2: 2x2 diagonal matrix [-1, 0; 0, -2]" << std::endl;

  std::vector<double> v2 = {-1.0, 0.0, -2.0};
  std::vector<CONE_IDX> idx2 = {0, 1, 2};

  ConeConstraintRaw cone2;
  cone2.cone = kConeSdp;
  cone2.idx = idx2.data();
  cone2.size = 3;

  ProxConeSdpCpu(cone2, v2.data());

  std::cout << "  Input:  [" << -1.0 << ", " << 0.0 << ", " << -2.0 << "]" << std::endl;
  std::cout << "  Output: [" << v2[0] << ", " << v2[1] << ", " << v2[2] << "]" << std::endl;
  std::cout << "  Expected: [0, 0, 0]" << std::endl;

  // Test 3: 2x2 matrix already in cone
  // Matrix: [2 0; 0 3]
  // After projection: [2, 0, 3] (unchanged)

  std::cout << "\nTest 3: 2x2 diagonal matrix [2, 0; 0, 3] (already PSD)" << std::endl;

  std::vector<double> v3 = {2.0, 0.0, 3.0};
  std::vector<CONE_IDX> idx3 = {0, 1, 2};

  ConeConstraintRaw cone3;
  cone3.cone = kConeSdp;
  cone3.idx = idx3.data();
  cone3.size = 3;

  ProxConeSdpCpu(cone3, v3.data());

  std::cout << "  Input:  [" << 2.0 << ", " << 0.0 << ", " << 3.0 << "]" << std::endl;
  std::cout << "  Output: [" << v3[0] << ", " << v3[1] << ", " << v3[2] << "]" << std::endl;
  std::cout << "  Expected: [2, 0, 3]" << std::endl;

  // Test 4: 3x3 matrix with mixed eigenvalues
  // Matrix: [2 1 0; 1 2 0; 0 0 -1]
  // Eigenvalues: 3, 1, -1
  // After projection: should have eigenvalues [3, 1, 0]

  std::cout << "\nTest 4: 3x3 matrix with mixed eigenvalues" << std::endl;

  std::vector<double> v4 = {2.0, 1.0, 0.0, 2.0, 0.0, -1.0};  // lower triangle
  std::vector<CONE_IDX> idx4 = {0, 1, 2, 3, 4, 5};

  ConeConstraintRaw cone4;
  cone4.cone = kConeSdp;
  cone4.idx = idx4.data();
  cone4.size = 6;

  std::cout << "  Input:  [" << v4[0] << ", " << v4[1] << ", " << v4[2]
            << ", " << v4[3] << ", " << v4[4] << ", " << v4[5] << "]" << std::endl;

  ProxConeSdpCpu(cone4, v4.data());

  std::cout << "  Output: [" << v4[0] << ", " << v4[1] << ", " << v4[2]
            << ", " << v4[3] << ", " << v4[4] << ", " << v4[5] << "]" << std::endl;

  return 0;
}
