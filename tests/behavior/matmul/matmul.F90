  program main

  integer, parameter :: N = 512
  real, dimension(N, N) :: A, B, C

  integer :: i, j, k
  real, volatile :: start, finish

  call init_array

  call cpu_time(start)

  do i = 1, N
    do j = 1, N
      C(i,j) = 0
      do k = 1, N
        C(i, j) = C(i, j) + A(i, k) * B(k, j)
      enddo
    enddo
  enddo

  call cpu_time(finish)

  print *, sum(C), minval(C), maxval(C)
  print *, 'time = ', finish - start

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

  end program main

