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

subroutine logexp(m_, n_, x_, y_, xy_)

use container

implicit none

integer, intent(in) :: m_, n_
real, intent(in) :: x_(m_, n_), y_(m_, n_)
real, intent(inout) :: xy_(m_, n_)

m = m_
n = n_

allocate(x(m, n))
allocate(y(m, n))
allocate(xy(m, n))

x = x_
y = y_

call container_log()
call container_exp()

xy_ = xy

deallocate(x)
deallocate(y)
deallocate(xy)

end subroutine logexp

