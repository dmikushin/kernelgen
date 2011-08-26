      subroutine cloudw( &
      
        qc1, qc2, qc3, &
        u2, u3, v2, v3, w2, w3, wsig2, wsig3, &
        
        wrk, difunqc, cond, auto, col, &
        pp, pp10, pp01, qcbx, qcby, qccc, &
        hk1, &
        
        ds0, ds1x, ds1y, ds12, taudra, hdamp, &
        dt, dtl, dx2, dy2, filt, vdampt, &
     
        nx, ny, ns, dx, dy, &
        nstep, idrmax, numsmo, iobptx, iobpty, &
     
        tfct, raylei, dohsmo, dovsmt, &
        rxmax, rymax)  
!
! time integration of the cloud water equation
!     
      implicit none
    
      real(8), dimension(0:nx+1,0:ny+1,0:ns+1) :: &
        qc1, qc2, qc3, &
        u2, u3, v2, v3, w2, w3, wsig2, wsig3, &
        wrk, &
        difunqc, auto, col, &
        cond
        
      real(8) :: func
        
      real(8), intent(inout) :: pp(0:nx+2,0:ny+2,1:4)
      
      real(8), intent(inout), dimension(0:nx+1,0:ny+1,1:4) :: pp10, pp01
     
      real(8), intent(inout) :: qcbx(1:2,0:ny+1,0:ns+1)
      real(8), intent(inout) :: qcby(0:nx+1,1:2,0:ns+1)
      real(8), intent(inout) :: qccc(1:2,1:2,0:ns+1)

      real(8), intent(inout) :: hk1(0:nx+1,0:ny+1)
      
      real(8), intent(inout), dimension(0:ns+1) :: &
        ds0, ds1x, ds1y, ds12, hdamp
      real(8), intent(inout), dimension(0:ns) :: taudra
      
      real(8), intent(inout) :: dt, dtl, dx2, dy2, filt, vdampt

      integer, intent(in) :: &
        nx, ny, ns, &
        nstep, idrmax, numsmo, iobptx, iobpty

      logical, intent(in) :: tfct, raylei, dohsmo, dovsmt
      
      real(8), intent(in) :: rxmax, rymax      

      real(8), allocatable, dimension(:,:,:) :: &
        work1, work2, work3, work4, work5, work6, work7, work8

      real(8) :: dx, dy

      integer :: ix, iy, is, k    

      if(tfct) then

        do is=0,ns
          do iy=3,ny
            do ix=1,nx+1
              if (ix / 2 .eq. 0) then
                qc3(ix,iy,is) = qc3(ix,iy-1,is) * pp(ix-1,iy,3)
              endif
              qc2(ix,iy,is) = qc2(ix-1+2,iy+1,is) * pp(ix+1,iy,2)
              qc1(ix,iy,is) = qc1(ix,iy-3,is) * pp(ix,iy,1) * pp(ix-1,iy,3)
            enddo
          enddo
        enddo

        allocate(work1(0:nx+1,0:ny+1,0:ns+1))
        allocate(work2(0:nx+1,0:ny+1,0:ns+1))
        allocate(work3(0:nx+1,0:ny+1,0:ns+1))
        allocate(work4(0:nx+1,0:ny+1,0:ns+1))
        allocate(work5(0:nx+1,0:ny+1,0:ns+1))
        allocate(work6(0:nx+1,0:ny+1,0:ns+1))
        allocate(work7(0:nx+1,0:ny+1,0:ns+1))
        allocate(work8(0:nx+1,0:ny+1,0:ns+1))
     
        call fct3d( &
          nx, ny, ns, &
          dx, dy, dt, ds0, ds1x, ds1y, &
          qc3, qc2, qc1, &
          u3, u2, v3, v2, wsig3, wsig2, &
          work1, work2, work3, work4, work5, work6, work7, work8)     
     
        deallocate(work1)
        deallocate(work2)
        deallocate(work3)
        deallocate(work4)
        deallocate(work5)
        deallocate(work6)
        deallocate(work7)
        deallocate(work8)

        do is=0,ns
          do iy=1,ny+1
            do ix=1,nx+1
              qc3(ix,iy,is) = qc3(ix,iy,is) / pp(ix,iy,3)
              qc2(ix,iy,is) = qc2(ix,iy,is) / pp(ix,iy,2)
              qc1(ix,iy,is) = qc1(ix,iy,is) / pp(ix,iy,1)
            enddo
          enddo
        enddo

        do is=1,ns-1
          do iy=2,ny
            do ix=2,nx
              qc3(ix,iy,is) = qc3(ix,iy,is) + dtl * (difunqc(ix,iy,is) + cond(ix,iy,is) - &
                auto(ix,iy,is) - col(ix,iy,is)) / pp(ix,iy,3)
            enddo
          enddo
        enddo
      else
!
! common leapfrog call
!
        call leapfrog( &
          nx, ny, ns, dx, dy, ds12, dtl, &
          u2, v2, wsig2, pp, &
          qc1, qc2, qc3)
!
! additional components
!     
        do is = 1, ns-1
          do iy = 2, ny
            do ix = 2, nx            
              qc3(ix,iy,is) = qc3(ix,iy,is) + (dtl * pp(ix,iy,2) * &
                (difunqc(ix,iy,is) + cond(ix,iy,is) - auto(ix,iy,is) - &
                col(ix,iy,is))) / pp(ix,iy,3)
            enddo
          enddo
        enddo
      endif
!
! surface cloud water flux (bulk parametrization)
!
! horizontal boundary conditions :
!
      call ptbc(nx, ny, ns, qc3, qcbx, qcby, qccc, iobptx, iobpty)

      if(raylei) then
        do is=1,idrmax
          do iy=1,ny+1
            do ix=1,nx+1
              qc3(ix,iy,is) = qc3(ix,iy,is) - dtl / taudra(is) * qc2(ix,iy,is)
            enddo
          enddo
        enddo
      endif

      do iy=1,ny+1
        do ix=1,nx+1
          qc3(ix,iy,ns) = func(nx, ny, ns, qc1, qc2, qc3, pp, qc3(ix,iy,ns-1))
          qc3(ix,iy,0) = qc3(ix,iy,1)
        enddo
      enddo

      if(dohsmo .and. mod(nstep,numsmo).eq.0) then
         call hsmooth(qc3, qc3, hk1, nx+1, ny+1, ns, 1, nx+1, 1, ny+1, 1, ns-1, hdamp, .FALSE.)
      endif
      
      if(dovsmt .and. mod(nstep,numsmo).eq.0) then
         call vsmooth(qc3, qc3, wrk, nx+1, ny+1, ns, 1, nx+1, 1, ny+1, 1, ns-1, vdampt, .FALSE.)
      endif

      if(.not.tfct) then
         call aselin(qc3, qc2, qc1, 0, nx+1, 0, ny+1, 0, ns, filt)
      endif

      call radbch( &
        rxmax, rymax, qccc, qcbx, qcby, qc3, qc2, qc1, &
        nx+1, ny+1, ns+1, dx, dy, dt, 1, ns-1, 1, nx+1, 1, ny+1)

      end subroutine cloudw
      
      subroutine cloudw1
      use alloc
      end
      
      function func(nx, ny, ns, qc1, qc2, qc3, pp, val)
      implicit none
      integer :: nx, ny, ns
      real(8) :: func, val
      real(8), dimension(0:nx+1,0:ny+1,0:ns+1) :: qc1, qc2, qc3       
      real(8), intent(inout) :: pp(0:nx+2,0:ny+2,1:4)
      
      integer:: ix, iy, is
       
      func = val
      
        do is=0,ns
          do iy=1,ny+1
            do ix=1,nx+1
              qc3(ix,iy,is) = qc3(ix,iy,is) * pp(ix,iy,3)
              qc2(ix,iy,is) = qc2(ix,iy,is) * pp(ix,iy,2)
              qc1(ix,iy,is) = qc1(ix,iy,is) * pp(ix,iy,1)
            enddo
          enddo
        enddo      
      
      end
      
      subroutine cloudw1
      use alloc
      end
      
      program main
      implicit none
      integer :: nx, ny, ns
      real(8) :: func, val
      real(8), allocatable, dimension(:,:,:) :: qc1, qc2, qc3       
      real(8), allocatable :: pp(:,:,:)
      
      integer:: ix, iy, is
       
      func = val
      
        do is=0,ns
          do iy=1,ny+1
            do ix=1,nx+1
              qc3(ix,iy,is) = qc3(ix,iy,is) * pp(ix,iy,3)
              qc2(ix,iy,is) = qc2(ix,iy,is) * pp(ix,iy,2)
              qc1(ix,iy,is) = qc1(ix,iy,is) * pp(ix,iy,1)
            enddo
          enddo
        enddo      
      
      end      
