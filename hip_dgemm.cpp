#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>
#include <hipblas.h>

int main(int argc, char *argv[]) {
  const int m = 46340;
  const int k = 46340;
  const int n = 46340;

  hipblasOperation_t op1 = HIPBLAS_OP_N;
  hipblasOperation_t op2 = HIPBLAS_OP_N;

  double alpha = 1.0;
  double beta = 0.0;

  // Host
  double *hA, *hB;
  hipHostMalloc((void**) &hA, m*k*sizeof(double));
  hipHostMalloc((void**) &hB, k*n*sizeof(double));

  // Device
  double *dA, *dB, *dC;
  hipMalloc((void**) &dA, m*k*sizeof(double));
  hipMalloc((void**) &dB, k*n*sizeof(double));
  hipMalloc((void**) &dC, m*n*sizeof(double));

  // Populate matrices on host
  for (int i=0; i<m*k; i++)
    hA[i] = ((double) i);
  for (int i=0; i<k*n; i++)
    hB[i] = ((double) i);

  hipStream_t s;
  hipStreamCreate(&s);

  hipblasHandle_t handle;
  hipblasStatus_t status;
  hipblasCreate(&handle);
  hipblasSetStream(handle,s);

  // Copy from host to device
  hipMemcpy(dA,hA,m*k*sizeof(double),hipMemcpyHostToDevice);
  hipMemcpy(dB,hB,n*k*sizeof(double),hipMemcpyHostToDevice);

  // Set leading dimensions of matrices
  int lda = (op1 == HIPBLAS_OP_N) ? m : k;
  int ldb = (op2 == HIPBLAS_OP_N) ? k : n;
  int ldc = m;

  // Perform DGEMM
  status = hipblasDgemm(handle, op1, op2, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);

  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::cout << "DGEMM unsuccessful!" << std::endl;
  }

  hipDeviceSynchronize();

  return 0;
}
