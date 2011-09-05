subroutine sincos_loop_1_kernelgen(nz, ny, nx, xy, x, y)
implicit none
interface
subroutine sincos_loop_1_kernelgen_blockidx_x(index, start, end) bind(C)
use iso_c_binding
integer(c_int) :: index
integer(c_int), value :: start, end
end subroutine
subroutine sincos_loop_1_kernelgen_blockidx_y(index, start, end) bind(C)
use iso_c_binding
integer(c_int) :: index
integer(c_int), value :: start, end
end subroutine
subroutine sincos_loop_1_kernelgen_blockidx_z(index, start, end) bind(C)
use iso_c_binding
integer(c_int) :: index
integer(c_int), value :: start, end
end subroutine
end interface
integer :: k
integer :: nz
integer :: j
integer :: ny
integer :: i
integer :: nx
real :: xy(nx, ny, nz)
real :: x(nx, ny, nz)
real :: y(nx, ny, nz)

#ifdef __CUDA_DEVICE_FUNC__
call sincos_loop_1_kernelgen_blockidx_z(k, 1, nz)
#else
do k = 1, nz
#endif

  !$acc do vector(2)
  
#ifdef __CUDA_DEVICE_FUNC__
call sincos_loop_1_kernelgen_blockidx_y(j, 1, ny)
#else
do j = 1, ny
#endif

    !$acc do vector(2)
    
#ifdef __CUDA_DEVICE_FUNC__
call sincos_loop_1_kernelgen_blockidx_x(i, 1, nx)
#else
do i = 1, nx
#endif

      xy(i, j, k) = sin(x(i, j, k)) + cos(y(i, j, k))
    
#ifndef __CUDA_DEVICE_FUNC__
enddo
#endif

  
#ifndef __CUDA_DEVICE_FUNC__
enddo
#endif


#ifndef __CUDA_DEVICE_FUNC__
enddo
#endif

end subroutine sincos_loop_1_kernelgen
