program dimension_test

integer :: a
dimension a(1:10)

do i = 1, 10
  a(i) = sqrt(real(i))
enddo

end program dimension_test

