subroutine logexp(m_, n_, x_, y_, xy_)

implicit none

integer, intent(in) :: m_, n_
real, intent(in), dimension(m_, n_) :: x_, y_
real, intent(inout) :: xy_(m_, n_)

integer :: m, n
real, allocatable, dimension(:, :) :: x, y
real, allocatable :: xy(:, :)

integer :: i, j

m = m_
n = n_

allocate(x(m, n))
allocate(y(m, n))
allocate(xy(m, n))

x = x_
y = y_

j = 1
do j = 1, n
  do i = 1, m
    xy(i, j) = log(x(i, j)) + exp(y(i, j))
  enddo
enddo

print *, 'Value of i after cycle = ', i
print *, 'Value of j after cycle = ', j

xy_ = xy

deallocate(x)
deallocate(y)
deallocate(xy)

end subroutine logexp

