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

MODULE GFORSCALE

USE ISO_C_BINDING

INTEGER, PARAMETER :: GFORSCALE_RUNMODE_HOST		= 1
INTEGER, PARAMETER :: GFORSCALE_RUNMODE_DEVICE_CUDA	= 2
INTEGER, PARAMETER :: GFORSCALE_RUNMODE_DEVICE_OPENCL	= 4

REAL(8) :: GFORSCALE_COMPARE_MAXDIFF = 1e-4

TYPE, BIND(C) :: GFORSCALE_KERNEL_CONFIG
  INTEGER(C_INT) :: RUNMODE, COMPARE
  INTEGER(C_INT) :: ILOOP, NLOOPS
  TYPE(C_PTR) :: ROUTINE_NAME
  TYPE(C_PTR) :: LAUNCH
  INTEGER(C_INT) :: NARGS, NMODSYMS
  TYPE(C_PTR) :: SPECIFIC
END TYPE GFORSCALE_KERNEL_CONFIG

TYPE, BIND(C) :: GFORSCALE_STATUS
  INTEGER(C_INT) :: VALUE, RUNMODE
END TYPE GFORSCALE_STATUS

CONTAINS

FUNCTION GFORSCALE_GET_LAST_ERROR()

  IMPLICIT NONE

  INTERFACE

    FUNCTION GFORSCALE_GET_LAST_ERROR_C() bind(C, name = "gforscale_get_last_error")
    USE ISO_C_BINDING
    IMPORT :: GFORSCALE_STATUS
    IMPLICIT NONE
    TYPE(GFORSCALE_STATUS) :: GFORSCALE_GET_LAST_ERROR_C
    END FUNCTION GFORSCALE_GET_LAST_ERROR_C

  END INTERFACE

  INTEGER :: GFORSCALE_GET_LAST_ERROR
  TYPE(GFORSCALE_STATUS) :: STATUS
  
  STATUS = GFORSCALE_GET_LAST_ERROR_C()
  GFORSCALE_GET_LAST_ERROR = STATUS%VALUE

END FUNCTION GFORSCALE_GET_LAST_ERROR

END MODULE GFORSCALE

