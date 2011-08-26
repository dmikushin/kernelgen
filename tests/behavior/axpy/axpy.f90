subroutine axpy(n, a, x, y)

implicit none

integer, intent(in) :: n
real, intent(in) :: a, x(n)
real, intent(inout) :: y(n)

integer :: i

loop: do i = 1, n
  y(i) = y(i) + a * x(i)
enddo loop

print *, 'Value of i after cycle = ', i

end subroutine axpy

