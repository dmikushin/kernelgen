subroutine sincos_loop_1_kernelgen(nz, ny, nx, xy, x, y)
implicit none
integer :: k
integer :: nz
integer :: j
integer :: ny
integer :: i
integer :: nx
real :: xy(nx, ny, nz)
real :: x(nx, ny, nz)
real :: y(nx, ny, nz)
do k = 1, nz
  !$acc do vector(2)
  do j = 1, ny
    !$acc do vector(2)
    do i = 1, nx
      xy(i, j, k) = sin(x(i, j, k)) + cos(y(i, j, k))
    enddo
  enddo
enddo
end subroutine sincos_loop_1_kernelgen
