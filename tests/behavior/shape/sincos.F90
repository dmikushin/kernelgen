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

subroutine sincos(nx, ny, nz, x, y, xy)

use shapes

implicit none

integer, intent(in) :: nx, ny, nz
real, intent(in) :: x(nx, ny, nz), y(nx, ny, nz)
real, intent(inout) :: xy(nx, ny, nz)

integer :: i, j, k

array1 = 1

!#define value1 array1(mod(i, 10) + 1, mod(j, 10) + 1, mod(k, 10) + 1)

do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      xy(i, j, k) = &
        sin(x(i, j, k)) * array1(mod(i, 10) + 1, mod(j, 10) + 1, mod(k, 10) + 1) + & !value1 + &
        cos(y(i, j, k)) * array2(mod(i, 10) + 1, mod(j, 10) + 1, mod(k, 10) + 1) !value1
    enddo
  enddo
enddo

print *, 'Value of i after cycle = ', i
print *, 'Value of j after cycle = ', j
print *, 'Value of k after cycle = ', k

end subroutine sincos

