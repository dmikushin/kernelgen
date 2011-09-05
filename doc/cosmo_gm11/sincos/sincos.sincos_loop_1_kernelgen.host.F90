subroutine sincos_loop_1_kernelgen_init_deps(config) bind(C)
USE KERNELGEN
type(kernelgen_kernel_config), bind(C) :: config
call kernelgen_kernel_init_deps(config)
end subroutine sincos_loop_1_kernelgen_init_deps
function sincos_loop_1_kernelgen_compare(verbose, maxdiff, nz, ny, nx, xy, x, y, nz_kernelgen, ny_kernelgen, nx_kernelgen, xy_kernelgen, x_kernelgen, y_kernelgen)
USE KERNELGEN
implicit none
integer :: sincos_loop_1_kernelgen_compare
integer(4) :: verbose
type(kernelgen_compare_maxdiff_t) :: maxdiff
real(8) :: ndiff, adiff
integer :: k
integer :: nz
integer :: nz_kernelgen
integer :: j
integer :: ny
integer :: ny_kernelgen
integer :: i
integer :: nx
integer :: nx_kernelgen
real :: xy(nx, ny, nz)
real :: xy_kernelgen(nx, ny, nz)
real :: x(nx, ny, nz)
real :: x_kernelgen(nx, ny, nz)
real :: y(nx, ny, nz)
real :: y_kernelgen(nx, ny, nz)

sincos_loop_1_kernelgen_compare = 0

if (((nz - nz_kernelgen)) .ne. 0 .or. ((nz - nz_kernelgen)) .ne. 0) &
  sincos_loop_1_kernelgen_compare = sincos_loop_1_kernelgen_compare + 1
if (((ny - ny_kernelgen)) .ne. 0 .or. ((ny - ny_kernelgen)) .ne. 0) &
  sincos_loop_1_kernelgen_compare = sincos_loop_1_kernelgen_compare + 1
if (((nx - nx_kernelgen)) .ne. 0 .or. ((nx - nx_kernelgen)) .ne. 0) &
  sincos_loop_1_kernelgen_compare = sincos_loop_1_kernelgen_compare + 1
ndiff = sqrt(sum((xy_kernelgen / xy - 1.0)**2) / size(xy))
adiff = max(abs(maxval(xy_kernelgen - xy)), abs(minval(xy_kernelgen - xy)))
if (verbose .ne. 0) then
print *, "ndiff(xy) = ", ndiff, "adiff(xy) = ", adiff
endif
if (((adiff .ne. adiff) .and. (ndiff .ne. ndiff)) .or. (ndiff .ge. maxdiff%single)) &
  sincos_loop_1_kernelgen_compare = sincos_loop_1_kernelgen_compare + 1
ndiff = sqrt(sum((x_kernelgen / x - 1.0)**2) / size(x))
adiff = max(abs(maxval(x_kernelgen - x)), abs(minval(x_kernelgen - x)))
if (verbose .ne. 0) then
print *, "ndiff(x) = ", ndiff, "adiff(x) = ", adiff
endif
if (((adiff .ne. adiff) .and. (ndiff .ne. ndiff)) .or. (ndiff .ge. maxdiff%single)) &
  sincos_loop_1_kernelgen_compare = sincos_loop_1_kernelgen_compare + 1
ndiff = sqrt(sum((y_kernelgen / y - 1.0)**2) / size(y))
adiff = max(abs(maxval(y_kernelgen - y)), abs(minval(y_kernelgen - y)))
if (verbose .ne. 0) then
print *, "ndiff(y) = ", ndiff, "adiff(y) = ", adiff
endif
if (((adiff .ne. adiff) .and. (ndiff .ne. ndiff)) .or. (ndiff .ge. maxdiff%single)) &
  sincos_loop_1_kernelgen_compare = sincos_loop_1_kernelgen_compare + 1

end function sincos_loop_1_kernelgen_compare
