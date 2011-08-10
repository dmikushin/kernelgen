 !
 ! KGen - the LLVM-based compiler with GPU kernels generation over C backend.
 !
 ! Copyright (c) 2011 Dmitry Mikushin
 !
 ! This software is provided 'as-is', without any express or implied warranty.
 ! In no event will the authors be held liable for any damages arising 
 ! from the use of this software.
 ! Permission is granted to anyone to use this software for any purpose, 
 ! including commercial applications, and to alter it and redistribute it freely,
 ! subject to the following restrictions:
 !
 ! 1. The origin of this software must not be misrepresented;
 ! you must not claim that you wrote the original software.
 ! If you use this software in a product, an acknowledgment
 ! in the product documentation would be appreciated but is not required.
 ! 2. Altered source versions must be plainly marked as such,
 ! and must not be misrepresented as being the original software.
 ! 3. This notice may not be removed or altered from any source distribution.
 !

subroutine logexp(m, n, x, y, xy)

implicit none

integer, intent(in) :: m, n
real, intent(in) :: x(m, n), y(m, n)
real, intent(inout) :: xy(m, n)

integer :: i, j, k

do j = 1, n
  ! This is just a comment
  do i = 1, m
    k = 1
    do while (k <= 1)
      xy(i, j) = log(x(i, j))
      k = k + 1
    enddo
  enddo
enddo

if (.true. .eqv. .false.) then

do j = 1, n
  ! This is just a comment
  do i = 1, m / 2
    do k = 1, 1
      xy(i, j) = log(x(i, j))
      write (*,*) 'hello'
    enddo
  enddo
  do i = m / 2 + 1, m
    xy(i, j) = log(x(i, j))
  enddo
enddo

endif

do j = 1, n
  xy(1, j) = xy(1, j) + exp(y(1, j))
  do i = 2, m
    do k = 1, 1
      xy(i, j) = xy(i, j) + exp(y(i, j))
      cycle
    enddo
  enddo
enddo

print *, 'Value of i after cycle = ', i
print *, 'Value of j after cycle = ', j

end subroutine logexp

