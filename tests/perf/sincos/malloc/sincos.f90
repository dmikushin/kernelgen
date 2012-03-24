subroutine sincos(nx, ny, nz, x, y, xy1, xy2)

implicit none

integer, intent(in) :: nx, ny, nz
real, intent(in), dimension(nx, ny, nz) :: x, y
real, intent(out), dimension(nx, ny, nz) :: xy1, xy2

integer :: i, j, k

! PGI Accelerator region
!$acc region
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      xy1(i,j,k) = sin(x(i,j,k)) + cos(y(i,j,k))
    enddo
  enddo
enddo
!$acc end region

! Control CPU implementation
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      xy2(i,j,k) = sin(x(i,j,k)) + cos(y(i,j,k))
    enddo
  enddo
enddo

! Compare results
print *, 'diff = ', maxval(xy1 - xy2)

end subroutine sincos	

