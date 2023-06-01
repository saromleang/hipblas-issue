program hip_dgemm

  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none

   integer, parameter :: m = 46340
   integer, parameter :: k = 46340
   integer, parameter :: n = 46340

  integer(kind(HIPBLAS_OP_N)), parameter :: op1 = HIPBLAS_OP_N
  integer(kind(HIPBLAS_OP_N)), parameter :: op2 = HIPBLAS_OP_N

  double precision, parameter ::  alpha = 1.0d0
  double precision, parameter ::  beta = 0.0d0

  ! Host
  double precision, allocatable, target, dimension(:) :: hA, hB, hC

  ! Device
  integer :: lda, ldb, ldc, size_a, size_b, size_c;
  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr, dC = c_null_ptr

  type(c_ptr) :: handle = c_null_ptr

  integer, parameter :: bytes_per_element = 8
  integer(c_size_t) :: Nabytes, Nbbytes, Ncbytes

  integer :: i

  size_a = m * k;
  size_b = n * k;
  size_c = m * n;

  Nabytes = size_a*bytes_per_element
  Nbbytes = size_b*bytes_per_element
  Ncbytes = size_c*bytes_per_element

  ! Host 
  allocate(hA(size_a))
  allocate(hB(size_b))
  allocate(hC(size_c))

  ! Device
  call hipCheck(hipMalloc(dA,Nabytes))
  call hipCheck(hipMalloc(dB,Nbbytes))
  call hipCheck(hipMalloc(dC,Ncbytes))

  ! Populate matrices on host
  do i=1, m*k
    hA(i)=dble(i)
  end do  

  do i=1, k*n
    hB(i)=dble(i)
  end do  

  call hipblasCheck(hipblasCreate(handle))

  ! Copy from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), Nabytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1)), Nbbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1)), Ncbytes, hipMemcpyHostToDevice))

  ! Set leading dimensions of matrices
  if (op1 == HIPBLAS_OP_N) then
     lda = m
  else
     lda = k
  end if

  if (op2 == HIPBLAS_OP_N) then
     ldb = k
  else
     ldb = n
  end if

  ldc = m

  ! Perform DGEMM
  call hipblasCheck(hipblasDgemm(handle, op1, op2, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc))

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(db))
  call hipCheck(hipFree(dc))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(ha)
  deallocate(hb)
  deallocate(hc)

end program hip_dgemm
