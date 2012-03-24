subroutine sincos(nx, ny, nz)

implicit none

integer :: nx, ny, nz
real, allocatable, dimension(:,:,:) :: x, y, xy1, xy2

integer :: i, j, k
real :: start, finish

allocate(x(nx, ny, nz))
allocate(y(nx, ny, nz))
allocate(xy1(nx, ny, nz))
allocate(xy2(nx, ny, nz))

! Fill input arrays with random values
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      call random_number(x(i,j,k))
      call random_number(y(i,j,k))
    enddo
  enddo
enddo

call cpu_time(start)

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

call cpu_time(finish)

print *, 'acc time = ', finish - start

call cpu_time(start)

! Control CPU implementation
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      xy2(i,j,k) = sin(x(i,j,k)) + cos(y(i,j,k))
    enddo
  enddo
enddo

call cpu_time(finish)

print *, 'cpu time = ', finish - start

! Compare results
print *, 'diff = ', maxval(xy1 - xy2)

deallocate(x)
deallocate(y)
deallocate(xy1)
deallocate(xy2)

end subroutine sincos

