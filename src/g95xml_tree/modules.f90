!
! this module includes all public varianles
!	
	module alloc
!
! grid size:
!
      integer :: &
        nall, nxy, nxs, nys, &
        jy, js, nall3, nallm, nallmm
        
      integer :: &
        isqv, isqc, isqr, &
        nx1qv, ny1qv, ns1qv, &
        nx1qc, ny1qc, ns1qc, &
        nx1qr, ny1qr, ns1qr
!
! dppref: reference state dp/dt
! psufref: reference state psuf
!
      real(8), allocatable, dimension(:,:,:) :: &
        difunq,qvs,cond,evap,difunqc,auto,col,vrain,difunqr
      real(8), allocatable, dimension(:,:) :: prec

      real(8), allocatable, dimension(:,:,:) :: &
        wk1, wk2, wk3, wk4, wk5, wk6, wk7, wk8, wk9

      real(8), allocatable, dimension(:,:,:) :: &
        tems, difunu, difunv, difunw, difunt, &
        s, thetav, difcoft
        
      real(8), allocatable, dimension(:,:) :: dppref, psufref
      
      real(8), allocatable, dimension(:,:) :: &
        tsuf, psufi, tsufi, phisui, deltaz, ptsoil, uvsuf, qvsoil, phisuf, &
        phig, hk1, hk2, hk3, ppgra2, pplap, gpplap
        
      real(8), allocatable, dimension(:,:,:) :: &
        pp, pp10, pp01, ppdx10, ppdx, dpdx10, ppdy01, ppdy, dpdy01
     
      real(8), allocatable, dimension(:,:) :: &
        dpp(:,:), hsufmax(:,:)
!
! boundary values of u, v, pp, pt - obtained by extrapolation (radiative boundary conditions)
!
      real(8), allocatable, dimension(:,:,:) :: &
        ubx, vbx, uby, vby, ucc, vcc, &
        ubx2, vbx2, uby2, vby2, ucc2, vcc2, &
        ptbx, ptby, ptcc, qvbx, qvby, qvcc, &
        qcbx, qcby, qccc, qrbx, qrby, qrcc, &
        pgradx, pgrads, pgrady
     
      real(8), allocatable, dimension(:,:) :: &
        ppbx, ppby, ppcc, &
        ppbx2, ppby2, ppcc2
    
      real(8), allocatable :: &
        qscc(:,:,:),qsbx(:,:,:),qsby(:,:,:)
!
! pt: perturbation potential temperature
! pts: reference state potential temperature
! tems: reference state temperature
! tsuf: temperature at the surface
! psuf: surface pressure
! ptop: pressure at the top of the model
! pp=psuf-ptop
! hsuf: height of the lower surface
!
! psufi,tsufi: psuf,tsuf in the initial state
!
! data for calculation of bulk transfer in the lowest hakf layer
!
! diffusion terms
!
! parameters for rayleigh relaxation in the absorption layer
!
      real(8), dimension(:), allocatable :: endrag, taudra, deltazt
      integer :: idrmax
!
! phi: perturbation geopotential
! phis: reference state geopotential
! phisuf: phis at the surface
! phig:
!
! pgradx,pgrady,pgrads: boundary values of pp gradient (auxiliary var)
!
! grid variables
!
      real(8), dimension(:), allocatable :: &
        sigma0, sigma1, &
        ds0, ds02, ds04, ds02a, ds04a, ds08a, ds1, ds12, ds14, &
        s1ds0, s0ds1, s0ds2a, s0ds4a, s1ds4a, &
        ds1x, ds1dx, ds1y, ds1dy, dtxs, dtys, dxys

      real(8) :: &
        dsng, dsr05, &
        dx2, dx4, dx8, dx05, dx025, dxx, dxdt2, dxt2, &
        dy2, dy4, dy8, dy05, dy025, dyy, dydt2, dyt2, &
        dxy, dxy025, dtxy, dtl

      integer :: nstep

      real(8), allocatable :: delka2(:)
!
! fcor: coriolis parameter
!
      real(8):: fcor
!
! utm coordinates
!
      real(8) :: xminutm, xmaxutm, yminutm, ymaxutm
!
! parameters for fft
!
      integer :: ifax(10), ifay(10)
      real(8), allocatable, dimension(:) :: &
        trigsx, trigsy, &
        sipcox, simcox, sipcoy, simcoy, &
        xlamb, ylamb
      real(8), allocatable, dimension(:,:) :: &
        xylamb, eigxy
!
! work space
!
! note: wm2 needed from moment to wsigbc
! note: uflux,vflux,wflux needed from flux to moment next time step
!
      real(8), allocatable :: theta(:,:,:)
      real(8), allocatable, dimension(:,:) :: &
        hkk1, hkk2, hkk3, hkk4, hkk5
      integer, allocatable, dimension(:,:) :: ihkk1, ihkk2

      real(8), allocatable, dimension(:,:,:) :: &
        rh, ri, &
        work1, work2, work3, work4, &
        work5, work6, work7, work8

      real(8), dimension(:), allocatable :: w1d1, w1d2
      
      integer, allocatable :: iworkk(:,:,:)
!
! air water options
!
       real(8) :: qif, qdrag
       integer :: ifqc, ifqr, ifblob
!
! soil radiation and precipitation and geostrophic wind
! rainacc: accumulated precipitation during the run
!
      real(8), allocatable, dimension(:,:) :: &
        xlake, pl, zonirr, tsmed, tslake, rainacc

      real(8), allocatable, dimension(:) :: xpl, xra, xrg

      integer :: nf, ndados, ifsoil, ifhle

      real(8) :: z0hz0

      real(8), allocatable, dimension(:,:) :: &
        temmax, temmin, evapor, ptforc, phi00s

      real(8), allocatable, dimension(:) :: ugeos, vgeos
      real(8), allocatable, dimension(:,:) :: xugeos, xvgeos

      real(8) :: fcori

      real(8), allocatable, dimension(:) :: tauspo, tauspoy

      real(8), allocatable, dimension(:) :: a1phi, a3phi, sine

      real(8), allocatable, dimension(:,:) :: a2phi, ctphi, dtphi, sines

      real(8), allocatable, dimension(:) :: a1pos3, a3pos3
      real(8), allocatable, dimension(:,:) :: a21pos3

      real(8), allocatable, dimension(:) :: aves

      real(8), allocatable :: errp(:,:)
      integer, allocatable :: idath(:,:)

      real(8), allocatable, dimension(:) :: &
        duul, duur, dvvl, dvvr, &
        phibx1, phiby1, phibx2, phiby2

      real(8), allocatable, dimension(:) :: &
        a1sphi, a2sphi, a3sphi, csphi, dsphi
      real(8), allocatable, dimension(:,:) :: hk11, hk12

      real(8), allocatable, dimension(:,:) :: dphsdx, dphsdy
      real(8), allocatable, dimension(:) :: hdamp

      real(8), allocatable, dimension(:,:) :: &
        ff, tbot, dlnpdx, dlnpdy, dlntdx, dlntdy, dpsudx, dpsudy
      real(8), allocatable, dimension(:,:,:) :: hk21, hk22

      real(8) :: dragm, dragt
      real(8), allocatable, dimension(:) :: zlevm, zlev
     
      integer(4) :: iflake, radpar, rat_par, nradcall, nlakecall, shortwave, longwave
      real(8) :: month, day, tau_rg
      real(8), dimension(:,:,:), allocatable :: Radheat, Srad, Lrad
      real(8), dimension(:,:), allocatable :: Srad_surf, Lrad_surf
      real(8), dimension(:,:), allocatable :: tsurf
!
! options:
!
! iophis = 1(hydrostatic) 2(poisson eq dirichlet) 3(poisson eq neuman)
! ioppuv = 1(u v define boundary dpp/dt) 2(rad.b.c. for pp define u v)
! iobxxx =
! iowlbc = 0(lateral b.c. for w from wsig, and wsig b.c. from contin.)
!          1(lateral b.c. for wsig from w, and w from u and v b.c.)
!
      logical :: tfct, inipt, raylei, olddra, mxspan, adjust

      real(8) :: dx, dy, dt, xl, yl, dt2
      
      integer :: xlcor, xrcor, ylcor, yrcor
      
      logical :: fluxxlcor, fluxylcor, fluxxrcor, fluxyrcor
      
      integer :: ntime

      integer :: &
        nupsur, nuppts, nupdif, &
        iophis, ioppuv, iowlbc, &
        iodify, iobphy, iobuux, iobuuy, &
        iobvvx, iobvvy, iobptx, iobpty, iobppx, iobppy
!
! mountain: iomtyp:1- 2d mountain; 2- 3d mountain
! ivarhs=0 mountain height is established at nstep=0 and is fixed
! ivarhs=1 mountain height grows linearly from nstep=0 to nstepgrow and stays fixed afterwards
! ivarhs=2 mountain height grows in steps
!
      integer :: iomtyp, ivarhs, nstepgrow, nstepflat, numsteps
      real(8) :: hmount, xmount, ymount, xspan, yspan, hxwid, hywid, hexp, xmountc, ymountc
!
! aselin time filter
!
      real(8) :: filt, ppfilt
!
! constants
! ha - the height corresponding to the surface pressure pa
!       
      real(8), parameter :: &
        r = 287., cp = 1005., g = 9.8066, hlat = 2.501e6, p00 = 1.e5, &
        rv = 461.51, hlatcp = hlat/cp, xk4 = 1.e-3, qco = 1.e-3, xnor = 1.e7, &
        rhol = 1.e3, xk1 = 1.506, dv = 2.26e-5, xkt = 2.43e-2, xniu = 1.51e-5, &
        rho0 = 1.23, xa = 842., xb = 0.8, sch = xniu / dv, &
        gm48 = 17.837870, gm38 = 4.694174, gm29 = 1.827355, &
        akapa = r / cp, gr = g / r, g2 = 2. * g, r05 = 0.5 * r, &
        gr2 = gr * 2., g4 = 4. * g, g05 = 0.5 * g, &
        omega = 7.292e-5, eps = 1.e-6, ttmin = 1.e-5

      real(8) :: xlatit, ptop, pa, ha
!
! iodif = 1 - constant dif.coef., 2 - deformation dependent
!
      logical :: dohsmo, dovsmo, perdam, dovsmt
      real(8) :: hdampm, hdampt, vdamp, vdampt
      integer :: numsmo

      real(8) :: rikey, difcof, uvdif, tsdif, difl
      integer :: iodif, nxsponge1, nxsponge2, nysponge1, nysponge2, ntaut, ntauspon
      real(8) :: cdcoef, taum, taut, zm, zt, tauspon
      
      real(8), parameter :: tdif = 1.d0, qsdif = 0.d0, qdif = 1.d0, dkapa = 0.8d0
!
! aux
!
      real(8) :: hh, declive, planalts, planalti
      
      integer :: npro
      real(8) :: t0ini, ug, vg, dugdz, dvgdz, psref, dzpro

      real(8), allocatable, dimension(:) :: zpro, fpro, fnpro, press0, press1, dpressdt
      real(8), allocatable, dimension(:,:) :: &
        thpro0, dthdz, dthdt, dthdzdt, &
        uspro0, dusdz, dusdt, dusdzdt, &
        vspro0, dvsdz, dvsdt, dvsdzdt, &
        qvspro0, dqvsdz, dqvsdt, dqvsdzdt, &
        thpro1, dthdz1, uspro1, dusdz1, &
        vspro1, dvsdz1, qvspro1, dqvsdz1, &
        tepro, presspro, rhopro, rhopro1
!
! ndat is the maximum number of points in the profile given
! ndat=max(ndth,ndus,ndvs,ndqvs)
!
      integer :: ndat

      real(8), allocatable, dimension(:,:) :: &
        thdat, usdat, vsdat, qvsdat, &
        zthdat, zusdat, zvsdat, zqvsdat, psdat, ptdat, pressdat, &
        tedat

      character(50) :: comlin(3)
      logical :: itepsuf

      integer :: ioreft, iorefq, iorefu, iorefv, ndth, ndus, ndvs, ndqvs
      real(8) :: pts0

      real(8) :: ug0, dugdz0, vg0, dvgdz0
!
! output of wave drag:
!
      logical :: drgprt, domom

      integer :: nlev
      real(8) :: dzlev
      
      integer :: iotdrg, iotduv
!
! parameters for radiative boundary conditions:
!
      real(8) :: pi,pi05
!
! old nh3dprec.inc
!
      real(8), parameter :: zero0 = 1.d-30, resmax = 1.d-12

      integer :: istep, kstep
      real(8), dimension(:,:), allocatable :: pequiv
!
! work variables for inpos3c/pos3c
!
      real(8), allocatable, dimension(:,:) :: emo, emi
      real(8), allocatable, dimension(:) :: vlam, e
!
! variables for out6
!
      real(8), allocatable, dimension(:,:) :: panom, pini

      integer ifluxcor
      real(8) dpmeddt

      end module alloc


      module refstate
!
! icorprof selects coordinates of the given profiles
! =0 model coordinates
! =1 utm coordinates
! =2 lon,lat - to convert to utm
!
      integer :: nprof, intrefst, mrefst, ifuprefs, icoorprof
      real(8), dimension(:), allocatable :: xprof, yprof, distinv
      real(8) :: tprof1, tprof2

      real(8), dimension(:,:,:), allocatable:: pweight, dweightx, dweighty
      
      logical :: realtime, forcedpp, forcepsuf, forcemflx, divrefstate

      end module refstate    