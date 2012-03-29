subroutine jacobi(m, n, nit, a, b, w0, w1, w2)

  implicit none

  integer, intent(in) :: m, n 			! matrix dimensions
  integer, intent(in) :: nit 			! the maximum number of iterations to perform

  real, intent(inout), dimension(m, n) :: a, b
  real, intent(in) :: w0, w1, w2

  integer :: i, j, it

  !$acc region
  do it = 1, nit
    do j = 2, n-1
      do i = 2, m-1
        a(i, j) = w0 *  b(i,   j  ) + &
                  w1 * (b(i-1, j  ) + b(i  , j-1) + b(i+1, j  ) + b(i  , j+1)) + &
                  w2 * (b(i-1, j-1) + b(i-1, j+1) + b(i+1, j-1) + b(i+1, j+1))
      enddo
    enddo
    do j = 2, n-1
      do i = 2, m-1
        b(i, j) = a(i, j)
      enddo
    enddo
  enddo
  !$acc end region
  
!  print *, maxval(a), minval(a)

end subroutine jacobi

