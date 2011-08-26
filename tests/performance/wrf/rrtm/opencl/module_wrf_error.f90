


MODULE module_wrf_error
  INTEGER :: wrf_debug_level = 0
  CHARACTER*256 :: wrf_err_message
!$OMP THREADPRIVATE (wrf_err_message)
CONTAINS

  LOGICAL FUNCTION wrf_at_debug_level ( level )
    IMPLICIT NONE
    INTEGER , INTENT(IN) :: level
    wrf_at_debug_level = ( level .LE. wrf_debug_level )
    RETURN
  END FUNCTION wrf_at_debug_level

  SUBROUTINE init_module_wrf_error
  END SUBROUTINE init_module_wrf_error

END MODULE module_wrf_error

SUBROUTINE wrf_message( str )
  IMPLICIT NONE
  CHARACTER*(*) str






  print*, TRIM(str)
END SUBROUTINE wrf_message


SUBROUTINE wrf_message2( str )
  IMPLICIT NONE
  CHARACTER*(*) str
  write(0,*) str



END SUBROUTINE wrf_message2

SUBROUTINE wrf_error_fatal3( file_str, line, str )
  USE module_wrf_error



  IMPLICIT NONE
  CHARACTER*(*) file_str
  INTEGER , INTENT (IN) :: line
  CHARACTER*(*) str
  CHARACTER*256 :: line_str

  write(line_str,'(i6)') line
  CALL wrf_message2( '-------------- FATAL CALLED ---------------' )

  IF ( line > 0 ) THEN
    CALL wrf_message( 'FATAL CALLED FROM FILE:  '//file_str//'  LINE:  '//TRIM(line_str) )
  ENDIF
  CALL wrf_message2( str )
  CALL wrf_message2( '-------------------------------------------' )
  CALL wrf_abort
END SUBROUTINE wrf_error_fatal3

SUBROUTINE wrf_error_fatal( str )
  USE module_wrf_error
  IMPLICIT NONE
  CHARACTER*(*) str
  CALL wrf_error_fatal3 ( ' ', 0, str )
END SUBROUTINE wrf_error_fatal



SUBROUTINE wrf_check_error( expected, actual, str, file_str, line )
  USE module_wrf_error
  IMPLICIT NONE
  INTEGER , INTENT (IN) :: expected
  INTEGER , INTENT (IN) :: actual
  CHARACTER*(*) str
  CHARACTER*(*) file_str
  INTEGER , INTENT (IN) :: line
  CHARACTER (LEN=512) :: rc_str
  CHARACTER (LEN=512) :: str_with_rc

  IF ( expected .ne. actual ) THEN
    WRITE (rc_str,*) '  Routine returned error code = ',actual
    str_with_rc = TRIM(str // rc_str)
    CALL wrf_error_fatal3 ( file_str, line, str_with_rc )
  ENDIF
END SUBROUTINE wrf_check_error
