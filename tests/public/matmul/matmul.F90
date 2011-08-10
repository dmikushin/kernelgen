  program main

  integer, parameter :: N = 1536
  real, dimension(N, N) :: A, B, C

  integer :: i, j, k

  call init_array

  do i = 1, N
    do j = 1, N
      C(i,j) = 0
      do k = 1, N
        C(i, j) = C(i, j) + A(i, k) * B(k, j)
      enddo
    enddo
  enddo

#ifdef PRINT
  call print_array
#endif

  contains

  subroutine init_array
  implicit none

  integer :: i, j

  do i = 1, N
    do j = 1, N
      A(i, j) = (1 + mod((i * j), 1024)) / 2.0
      B(i, j) = (1 + mod((i * j), 1024)) / 2.0
    enddo
  enddo

  end subroutine init_array



  subroutine print_array
  implicit none

  integer :: i, j

  do i = 1, N
    do j = 1, N
      write(*, *) C(i, j)
      if (mod(j, 80) .eq. 79) print *
    enddo
    print *
  enddo

  end subroutine print_array

  end program main

