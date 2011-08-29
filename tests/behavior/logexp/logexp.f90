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

integer :: i, j

do j = 1, n
  do i = 1, m
    xy(i, j) = log(x(i, j)) + exp(y(i, j))
  enddo
enddo

print *, 'Value of i after cycle = ', i
print *, 'Value of j after cycle = ', j

end subroutine logexp
