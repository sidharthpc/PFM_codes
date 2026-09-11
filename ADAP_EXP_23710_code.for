!=======================================================================
!  PHASE-FIELD FRACTURE UEL/UMAT FOR ORTHOTROPIC FGM
!=======================================================================
!  Author       : Sidharth P. C & B. N. Rao
!  Affiliation  : Indian Institute of Technology Madras (IIT Madras)
!
!  DESCRIPTION
!  -----------
!  This is a set of Abaqus user subroutines (UEL, UMAT, UEXTERNALDB and
!  supporting utility routines) implementing a phase-field model for
!  brittle/anisotropic fracture in functionally graded materials (FGM),
!  where material properties vary spatially (here, along the Y direction).
!
!  Two coupled 4-noded quadrilateral user elements are defined:
!    JTYPE = 1  ->  PHASE-FIELD element
!                   Solves the phase-field (damage) evolution equation
!                   using the crack driving (elastic strain) energy
!                   history stored from the displacement element.
!    JTYPE = 2  ->  DISPLACEMENT element
!                   Solves the mechanical equilibrium equations using
!                   an orthotropic/anisotropic constitutive law, with
!                   stiffness degraded by the current phase-field value
!                   (standard phase-field degradation function).
!
!  Key features implemented:
!    - Functionally graded material properties: elastic moduli (E1, E2,
!      shear modulus) vary exponentially with the Y-coordinate to model
!      a graded microstructure.
!    - Anisotropic/orthotropic elasticity with a user-defined fiber
!      orientation angle (THETA), including a "hybrid anisotropic"
!      stiffness formulation for the phase-field equation (matrix AMAT
!      built from an isotropic part plus a directional correction).
!    - Volumetric-deviatoric / tension-compression (spectral)
!      decomposition of the strain energy, so that damage is driven
!      mainly by tensile/positive strain energy (helper subroutine
!      KDDSDDE performs the spectral split of stresses/stiffness).
!    - EXPONENTIAL (GRADED) FINITE ELEMENT SHAPE FUNCTIONS:
!      Subroutine KSHAPEFCN does not always use the standard bilinear
!      Q4 shape functions. When an element is flagged for refinement
!      (rcflag > 0.5), it constructs a family of one-dimensional
!      exponential blending functions (arrays RNEB/DNEB, built from
!      terms like (exp(-dli*(1+g)/4) - 1)/(exp(-dli/2) - 1) and their
!      derivatives) along each local edge/direction. These exponential
!      functions bias the mapping so that the effective shape functions
!      and their derivatives are compressed/stretched non-uniformly
!      across the element, concentrating resolution toward one side of
!      the element (selected via the rav/rbv corner-direction flags).
!      The resulting nodal blending functions (dNE) and their global
!      derivatives (dNdzEXP) are combined to build a modified element
!      shape-function set (distinct from the standard bilinear AN/dNdxi
!      used for geometry mapping) that is then used for the phase-field
!      interpolation and its gradient in the stiffness/residual
!      assembly. A small residual (dnerror) is redistributed among the
!      four nodal exponential shape functions to enforce partition of
!      unity. When rcflag < 0.5, this exponential scheme is bypassed
!      and the standard bilinear shape functions are used instead
!      (dNE = AN, dNdzEXP = dNdxi).
!    - Adaptive integration scheme: elements switch between a standard
!      2x2 Gauss quadrature and a refined higher-order Gauss-Legendre
!      quadrature (subroutine KINPTDATA, up to 10x10 points) when the
!      exponential shape-function refinement above is active.
!    - KORIENT decides, per element and per iteration, whether an
!      element should switch into the refined/exponential mode (based
!      on the sum of nodal phase-field DOFs, PHISWITCH threshold) and
!      determines the local orientation flags (rav, rbv) that steer
!      which corner/direction the exponential grading is biased toward.
!    - History variables (phase field, driving energies, stresses,
!      strains, orientation/refinement flags, etc.) are stored in the
!      common block /KUSER/ (array USRVAR) and passed between the
!      phase-field and displacement elements, which share element
!      numbering offset by N_ELEM.
!    - UEXTERNALDB periodically writes the current nodal phase-field
!      distribution (x, y, z, phi) for all elements to a CSV file
!      ("Phi.csv") in the job's output directory, once per unit time
!      increment, for post-processing/visualization.
!    - UMAT is a placeholder ("dummy") elastic material used only to
!      make solution-dependent state variables available at material
!      points other than the UEL integration points.
!
!
!  CHANGELOG (see also file header below)
!  ---------------------------------------
!    - Stable code for linear/adaptive integration points
!    - CSV output updated more frequently
!    - Volumetric and spectral decomposition of the strain tensor
!    - Hybrid anisotropic phase-field stiffness formulation
!    - Stable orthotropic material formulation
!    - Y-direction FGM property grading
!=======================================================================


      module kvisual
      implicit none
      real*8 PhiiV(200000,12)
      integer nelem,INTDATA(200000,2),nlines
      integer track(70000,3)
      PARAMETER N_ELEM=23710
      save
      end module
      
      subroutine uexternaldb(lop,lrestart,time,dtime,kstep,kinc)  
      use kvisual
      include 'aba_param.inc'
      integer :: i, j, nrow, ncol
      real *8 zr,quot,remain,time(2),tvar
C$$$$$$       character(len=20) :: filename
      character xoutdir*255, xfname*80
      character dmkname*255, fnamex*80
      integer k9,k8,error,io
      real*8, allocatable :: cs(:,:)

c
      lxfname = 0
      lxoutdir = 0
      xfname  =' '
      xoutdir  =' '

      zr=0.d0
      tvar=nint(time(2)*10000.d0)/10000.d0
      call getjobname(xfname,lxfname)     !input file name
      call getoutdir(xoutdir,lxoutdir)    !output directory
      quot=int(tvar/1.0d0)
      remain=tvar-(1.0d0*quot)
      
      if(lop.eq.2) then
         if (remain.eq.0.d0) then
             PRINT*,'UPDATING CSV'
             fnamex=dmkname(xfname(1:lxfname),xoutdir(1:lxoutdir),
     & 'Phi.csv')
         
           ! Open file for writing
           open(unit=95,file=fnamex,status='unknown',form='formatted')
         
           ! Write header row
           write(95, *) "x, y, z, phi"
           
             do k8=1,nelem
               do k7=1,4
                 k6=(k7-1)*3+1
                 k5=(k7-1)*3+2
                 k4=(k7-1)*3+3
                 write(95, '(F5.3, A)', advance="no") PhiiV(k8,k6), ","
                 write(95, '(F5.3, A)', advance="no") PhiiV(k8,k5), ","
                 write(95, '(F5.3, A)', advance="no") zr, ","
                 write(95, '(F5.3)', advance="yes") PhiiV(k8,k4)
               enddo
             end do
                 ! Close file
          close(95)
          else
          endif
      endif
	  
C$$$$$$       if(lop.eq.0) then
C$$$$$$ C
C$$$$$$       fnamex=dmkname(xfname(1:lxfname),xoutdir(1:lxoutdir),'track.txt')
C$$$$$$       nlines=5537
C$$$$$$ C$$$$$$       open(unit=97,file=fnamex,status='unknown',form='formatted')
C$$$$$$ C$$$$$$           do
C$$$$$$ C$$$$$$             read(97,*,iostat=io)
C$$$$$$ C$$$$$$             if (io/=0) EXIT
C$$$$$$ C$$$$$$               nlines=nlines+1
C$$$$$$ C$$$$$$           end do 
C$$$$$$ C$$$$$$           close(97)
C$$$$$$       allocate(cs(nlines,3))
C$$$$$$       open(unit=96,file=fnamex,status='old')
C$$$$$$           do i=1,nlines
C$$$$$$             read(96,*,iostat=io)track(i,1),track(i,2),track(i,3)
C$$$$$$           end do 
C$$$$$$       print*,nlines,'Elements marked for Correction'
C$$$$$$ C
C$$$$$$       endif
C$$$$$$       close(96)
	  
      return
      end
      
      character*(*) function dmkname(fname,dname,exten)
C
      character*(*) fname,dname,exten
C     fname  I   jobname
C     dname  I   directory
C     exten  I   extension
C     dmkname O directory/jobname.exten

      ltot = len(fname)
      lf = 0
      do k1 = ltot,2,-1
        if (lf.eq.0.and.fname(k1:k1).ne.' ')  lf = k1
      end do

      ltot = len(dname)
      ld = 0
      do k1 = ltot,2,-1
        if (ld.eq.0.and.dname(k1:k1).ne.' ')  ld = k1
      end do

      ltot = len(exten)
      le = 0
      do k1 = ltot,2,-1
        if (le.eq.0.and.exten(k1:k1).ne.' ')  le = k1
      end do

      if ((lf + ld + le) .le. len(dmkname)) then
        dmkname = dname(1:ld)//'/'//fname(1:lf)
        ltot = ld + lf + 1
        if ( le.gt.0) then
           dmkname = dmkname(1:ltot)//exten(1:le)
        end if
      end if
C
      return
      end

      
      SUBROUTINE UEL(RHS,AMATRX,SVARS,ENERGY,NDOFEL,NRHS,NSVARS,
     1     PROPS,NPROPS,COORDS,MCRD,NNODE,U,DU,V,A,JTYPE,TIME,DTIME,
     2     KSTEP,KINC,JELEM,PARAMS,NDLOAD,JDLTYP,ADLMAG,PREDEF,
     3     NPREDF,LFLAGS,MLVARX,DDLMAG,MDLOAD,PNEWDT,JPROPS,NJPROP,
     4     PERIOD)

      use kvisual
C     ==================================================================
      INCLUDE 'ABA_PARAM.INC'
C     ==================================================================
      PARAMETER(ZERO=0.D0,ONE=1.D0,MONE=-1.D0,TWO=2.D0,THREE=3.D0,
     1 TOLER=1.0D-8,FOUR=4.D0,RP25 = 0.25D0,HALF=0.5D0,SIX=6.D0,
     2 NSTVTO=4,NSTVTT=16,NSTV=24,PHISWITCH=0.5D0)
C     ==================================================================
C     Initialization for all the element types
C     ==================================================================
      DIMENSION RHS(MLVARX,1),AMATRX(NDOFEL,NDOFEL),
     1     SVARS(NSVARS),ENERGY(8),PROPS(NPROPS),COORDS(MCRD,NNODE),
     2     U(NDOFEL),DU(MLVARX,1),V(NDOFEL),A(NDOFEL),TIME(2),
     3     PARAMS(3),JDLTYP(MDLOAD,*),ADLMAG(MDLOAD,*),
     4     DDLMAG(MDLOAD,*),PREDEF(2,NPREDF,NNODE),LFLAGS(*),
     5     JPROPS(*)
     
       INTEGER I,J,L,K,K1,K2,K3,K4,IX,IY,NINPT,OLDNINPT,NINPTC,NINP
      
       REAL*8 AINTW(100),XII(100,2),XI(2),dNdxi(NNODE,2),
     1 VJACOB(2,2),dNdx(NNODE,2),VJABOBINV(2,2),AN(4),BP(2,NDOFEL),
     2 DP(2),SDV(NSTV),BB(3,NDOFEL),CMAT(3,3),EPS(3),STRESS(3),
     3 VNI(2,NDOFEL),ULOC(2),PHASENOD(NNODE)
     
       Real*8 DTM,THCK,HIST,CLPAR,GCPAR,E1x,E2x,NU12,NU21,PARK,ENG
       Real*8 eg,elam,bk,trE,Edev(3),EdevS,trEp,trEn,psip,psin
       Real*8 xc,yc,dh(4),delta(4),Iinv(2,2)
       Real*8 coord16(100,2),wght(100)
       Real*8 dNE(4),dNdzEXP(4,2)
       Real*8 rcflag,rav,rbv,rcflag2,rav2,rbv2,HISTA
       Real*8 DNDXH(4,2),UU(8),HIST2,HIST3,ENGN2,ENGN3,HISTN2,HISTN3
       Real*8 EPS2(4),PS(3),PV(3,3),ddsdde(4,4),STRESS4(4)
       Real*8 trp1,trn1,trp2,trn2,psips,psins
       Real*8 AMAT(2,2),Abeta,Imat(2,2),theta,Across(2,2),BTA(4,2)
       Real*8 Gbulk,Oalpha,Obeta,Ogamma,Qmat(3,3),EMOD,NU
       Real*8 BTB(2,4),CCMAT(3,3),mrkr(1,3)
       Real*8 psif,psim1,psim2,Gc1,Gc2,Gc3,psin1,psin2
       real*4 Tmat(3,3),Tinv(3,3),Rmat(3,3),Rinv(3,3)
	   integer k11
 
       COMMON/KUSER/USRVAR(N_ELEM,NSTV,100)
C
C     ==================================================================
C     ******************************************************************
C     Constructing elemet TYPE 1 (phase)
C     ******************************************************************
C     ==================================================================
       Oalpha=0.5d0
       Obeta=0.4d0
       Ogamma=0.3d0
       theta=45.d0*(3.142d0/180.d0)
 
       
       nelem=N_ELEM
       AINTW=ZERO
c      OLDNINPT=2
       rcflag=0.0d0
       rav=0.d0
       rbv=0.d0
       NINP=2
      
C$$$$$$       do k11=1,nlines  
C$$$$$$         if (jelem.eq.track(k11,1)) then
C$$$$$$             mrkr(1,1)=1.d0
C$$$$$$             mrkr(1,2)=track(k11,2)
C$$$$$$             mrkr(1,3)=track(k11,3)
C$$$$$$         endif
C$$$$$$       end do 

C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,mrkr(1,1),mrkr(1,2),mrkr(1,3)
C$$$$$$         endif
     
       IF (JTYPE.EQ.ONE) THEN
C     ==================================================================
C     Time an iteration variables
C     ==================================================================
       TIMEZ=USRVAR(JELEM,21,1)
       IF (TIMEZ.LT.TIME(2)) THEN
        USRVAR(JELEM,21,1)=TIME(2)
        USRVAR(JELEM,22,1)=ZERO
       ELSE
        USRVAR(JELEM,22,1)=USRVAR(JELEM,22,1)+ONE
       ENDIF
       STEPITER=USRVAR(JELEM,22,1)

       IF (JELEM.EQ.1) THEN
         IF (STEPITER.EQ.ZERO) THEN
           PRINT*,'TIME=',TIME(2)
         ENDIF
       ENDIF 
      
C     ==================================================================
C     BUILDING RCFALG INTO USERVAR
C     ==================================================================
c      OLDNINPT=INTDATA(JELEM,1)
      
      IF (TIME(2).LT.1.0d0) THEN
        INTDATA(JELEM,1)=4
        rcflag=0.d0 
        NINP=2
      ELSE
        IF (STEPITER.EQ.ZERO) THEN
        CALL KORIENT(U,PHISWITCH,rav2,rbv2,rcflag2,jelem,mrkr,NINPTC)
        INTDATA(JELEM,2)=NINPTC
        USRVAR(JELEM,23,1)=rcflag2
        USRVAR(JELEM,23,2)=rav2
        USRVAR(JELEM,23,3)=rbv2
        rcflag=rcflag2
        rav=rav2
        rbv=rbv2
        NINP=NINPTC
        INTDATA(JELEM,1)=INTDATA(JELEM,2)
        ELSE
        NINP=INTDATA(JELEM,1)
        rcflag=USRVAR(JELEM,23,1)
        rav=USRVAR(JELEM,23,2)
        rbv=USRVAR(JELEM,23,3)
        ENDIF     
        
      ENDIF
      
C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,rcflag,rav,rbv,NINP
C$$$$$$         endif
C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,rcflag,NINP
C$$$$$$         endif

      
C     ==================================================================
C     Material parameters
C     ==================================================================
      
       CLPAR = PROPS(1)
       THCK = PROPS(2)
       kexp=PROPS(3)
       E1o=PROPS(4) 
       E2o=PROPS(5) 
       G12o=PROPS(6)
       XNU12=PROPS(7)
       Gc1=PROPS(8)
       Gc2=PROPS(9)
       Gc3=PROPS(10)
C     ==================================================================
C     Initial preparations
C     ==================================================================
       DO K1 = 1, NDOFEL                      
        DO KRHS = 1, NRHS
         RHS(K1,KRHS) = ZERO
        END DO
        DO K2 = 1, NDOFEL
         AMATRX(K2,K1) = ZERO
        END DO
       END DO
C     ==================================================================
C     Local weights
C     ================================================================== 
       
       IF (rcflag.lt.0.5d0) then
         do I=1,4
           AINTW(I)=1.d0
         enddo
       elseif (rcflag.gt.0.5d0) then
         call kinptdata(wght,coord16,NINP)
         do I=1,100
           AINTW(I)=wght(I)
         enddo
       endif

C$$$$$$       dh(1)=((COORDS(1,2)-COORDS(1,1))**2.d0+(COORDS(2,2)-
C$$$$$$      1 COORDS(2,1))**2.d0)**0.5d0
C$$$$$$       dh(2)=((COORDS(1,3)-COORDS(1,2))**2.d0+(COORDS(2,3)-
C$$$$$$      1 COORDS(2,2))**2.d0)**0.5d0
C$$$$$$       dh(3)=((COORDS(1,4)-COORDS(1,3))**2.d0+(COORDS(2,4)-
C$$$$$$      1 COORDS(2,3))**2.d0)**0.5d0
C$$$$$$       dh(4)=((COORDS(1,4)-COORDS(1,1))**2.d0+(COORDS(2,4)-
C$$$$$$      1 COORDS(2,1))**2.d0)**0.5d0
     
       DO i=1,4
       delta(i)=0.002/CLPAR
       END DO

C$$$$$$          if (jelem.eq.104.d0) then
C$$$$$$           print*,mrkr
C$$$$$$           print*,NINP
C$$$$$$          endif 

C     ==================================================================
C     Calculating properties at each integration point
C     ==================================================================
       NINPT=NINP*NINP
       DO INPT=1,NINPT
C     Initializing solution dependent variables (phase,history)
        DO I=1,NSTVTO
          SDV(I)=SVARS(NSTVTO*(INPT-1)+I)
        END DO
C
C     Shape functions and local derivatives
        CALL kshapefcn(INPT,delta,rav,rbv,rcflag,AN,dNdxi,dNE,dNdzEXP,
     1   jtype,NINPT)
C$$$$$$ 
C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,rcflag,rav,rbv
C$$$$$$           print*,'dNdxi',dNdxi
C$$$$$$           print*,'dNdzEXP',dNdzEXP
C$$$$$$         endif

C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,rcflag,rav,rbv,NINP
C$$$$$$         endif
       
!       coordinates in global
        xc=COORDS(1,1)*AN(1)+COORDS(1,2)*AN(2)+COORDS(1,3)*AN(3)+
     1     COORDS(1,4)*AN(4)
        yc=COORDS(2,1)*AN(1)+COORDS(2,2)*AN(2)+COORDS(2,3)*AN(3)+
     1     COORDS(2,4)*AN(4)

c       call kprofin(props,xc,yc,Es,xnus,Gcs)
       E1x=E1o*exp(Oalpha*yc)
       E2x=E2o*exp(Obeta*yc)
       Gbulk=G12o*exp(Ogamma*yc)
       NU12=XNU12
       NU21=NU12*E2x/E1x
C$$$$$$        GCPAR=Gc2
       GCPAR=1.d0
C$$$$$$        GCPAR=GCPAR/(1.d0+0.25d0*(0.005d0/CLPAR))
       
C     Jacobian
        DO I = 1,2
         DO J = 1,2
          VJACOB(I,J) = ZERO
          DO K = 1,NNODE
           VJACOB(I,J) = VJACOB(I,J) + COORDS(I,K)*dNdxi(K,J)
          END DO
         END DO
        END DO
C        

        DTM = ZERO
        DTM = VJACOB(1,1)*VJACOB(2,2)-VJACOB(1,2)*VJACOB(2,1)
        IF (DTM.LT.ZERO) THEN
         WRITE(7,*) 'Negative Jacobian',DTM,'| NINPT=',NINPT,
     1  '| rcflag,rav,rbv=',rcflag,rav,rbv,STEPITER
         CALL XIT	
        END IF
C     Inverse of Jacobian
        VJABOBINV(1,1)=VJACOB(2,2)/DTM
        VJABOBINV(1,2)=-VJACOB(1,2)/DTM
        VJABOBINV(2,1)=-VJACOB(2,1)/DTM
        VJABOBINV(2,2)=VJACOB(1,1)/DTM
C        
C     Derivatives of shape functions respect to global ccordinates
        DO K = 1,NNODE
         DO I = 1,2
          dNdx(K,I) = ZERO
          DO J = 1,2
             dNdx(K,I) = dNdx(K,I) + dNdzEXP(K,J)*VJABOBINV(J,I)
          END DO
         END DO
        END DO

        DO K = 1,NNODE
         DO I = 1,2
          DNDXH(K,I) = ZERO
          DO J = 1,2
             DNDXH(K,I) = DNDXH(K,I) + dNdxi(K,J)*VJABOBINV(J,I)
          END DO
         END DO
        END DO
C
C     Calculating B matrix (B=LN)
       DO INODE=1,NNODE
        BP(1,INODE)=dNdx(INODE,1)
        BP(2,INODE)=dNdx(INODE,2)
       END DO
C
C     ==================================================================
C     Nodal phase-field
C     ==================================================================
        PHASE=ZERO
        DPHASE=ZERO
        DO I=1,4
         PHASE=PHASE+dNE(I)*U(I)
        END DO
        DO I=1,4
         DPHASE=DPHASE+dNE(I)*DU(I,1)
        END DO

        IF (PHASE.GT.1.D0) THEN
          PHASE=1.D0
C$$$$$$           Print*,'Phase limited in TP1'
        ENDIF
C        
        IF (STEPITER.EQ.ZERO) THEN
          SDV(1)=PHASE-DPHASE
        ELSE
          SDV(1)=PHASE
        ENDIF
C   
C     Gradient
        DO I=1,2
         DP(I)=ZERO
        END DO
        DO I=1,2
         DO J=1,4
          DP(I)=DP(I)+BP(I,J)*U(J)
         END DO
        END DO
C
C     ==================================================================
C     Calculating elastic ENERGY history
C     ==================================================================
        IF (STEPITER.EQ.ZERO) THEN
         ENGN=USRVAR(JELEM,13,INPT)
         ENGN2=USRVAR(JELEM,15,INPT)
         ENGN3=USRVAR(JELEM,16,INPT)
        ELSE
         ENGN=USRVAR(JELEM,18,INPT)
         ENGN2=USRVAR(JELEM,19,INPT)
         ENGN3=USRVAR(JELEM,20,INPT)
        ENDIF
C        
        HISTN=USRVAR(JELEM,18,INPT)
        HISTN2=USRVAR(JELEM,19,INPT)
        HISTN3=USRVAR(JELEM,20,INPT)
        
        IF (ENGN.GT.HISTN) THEN
         HIST=ENGN
        ELSE
         HIST=HISTN
        ENDIF
        IF (ENGN2.GT.HISTN2) THEN
         HIST2=ENGN2
        ELSE
         HIST2=HISTN2
        ENDIF
        IF (ENGN3.GT.HISTN3) THEN
         HIST3=ENGN3
        ELSE
         HIST3=HISTN3
        ENDIF

        
        SDV(2)=HIST
        SDV(3)=HIST2
        SDV(4)=HIST3

C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,HIST,HIST2,HIST3
C$$$$$$         endif

        HISTA=(HIST/Gc2)+(HIST2/Gc2)+(HIST3/Gc2)
        HIST=HISTA
C$$$$$$         HIST=HIST/(Gc2)

C     ==================================================================
C     Calculating  ENERGY history FROM LAST VALUE OF DISPLACEMENTS
C     ==================================================================
C$$$$$$       DO I=1,8
C$$$$$$         UU(I)=USRVAR(JELEM,20,I)
C$$$$$$       ENDDO
C$$$$$$ 
C$$$$$$       IF (STEPITER.EQ.ZERO) THEN
C$$$$$$         IF (TIME(2).GT.1.0D0) THEN
C$$$$$$          IF (NINPT.GT.OLDNINPT) THEN
C$$$$$$              CALL HFINDER(UU,EMOD,ENU,AN,DNDXH,HISTA)
C$$$$$$              PRINT*,time(2),JELEM,OLDNINPT,NINPT,HISTA
C$$$$$$              HIST=HISTA
C$$$$$$          ENDIF
C$$$$$$         ENDIF
C$$$$$$       ENDIF

C     ==================================================================
C     Calculating A matrix
C     ==================================================================
      DO I=1,2
        DO J=1,2
          Imat(I,J)=ZERO
        END DO
      END DO
      DO I=1,2
        DO J=1,2
          Across(I,J)=ZERO
        END DO
      END DO
      DO I=1,2
        DO J=1,2
          AMAT(I,J)=ZERO
        END DO
      END DO
      Abeta=40.0d0
      Imat(1,1)=1.d0
      Imat(2,2)=1.d0
      Imat(1,2)=0.d0
      Imat(2,1)=0.d0
      Across(1,1)=sin(theta)*sin(theta)
      Across(1,2)=-cos(theta)*sin(theta)
      Across(2,1)=-sin(theta)*cos(theta)
      Across(2,2)=cos(theta)*cos(theta)
 
      AMAT=Imat+Abeta*(Imat-Across)
c       AMAT=Imat
      
C     ==================================================================
C     Calculating element stiffness matrix
C     ==================================================================
      BTA=matmul(transpose(BP),AMAT)
      BTB=transpose(BTA)
         
        DO I=1,NNODE
         DO K=1,NNODE
          DO J=1,2
           AMATRX(I,K)=AMATRX(I,K)+BTB(J,I)*BP(J,K)*DTM*
     1      THCK*GCPAR*CLPAR*AINTW(INPT)
          END DO
          AMATRX(I,K)=AMATRX(I,K)+dNE(I)*dNE(K)*DTM*THCK*
     1     AINTW(INPT)*(GCPAR/CLPAR+TWO*HIST)
         END DO
        END DO
C        
C     ==================================================================
C     Internal forces (residual vector)
C     ==================================================================
        DO I=1,NDOFEL
         DO J=1,2
           RHS(I,1)=RHS(I,1)-BTB(J,I)*DP(J)*GCPAR*CLPAR*
     1      AINTW(INPT)*DTM*THCK
         END DO
         RHS(I,1)=RHS(I,1)-dNE(I)*AINTW(INPT)*DTM*THCK*
     1    ((GCPAR/CLPAR+TWO*HIST)*PHASE-TWO*HIST)
        END DO
C
C     ==================================================================
C     Uploading solution dep. variables
C     ==================================================================
        DO I=1,NSTVTO
         SVARS(NSTVTO*(INPT-1)+I)=SDV(I)
         USRVAR(JELEM,I+NSTVTT,INPT)=SVARS(NSTVTO*(INPT-1)+I)
        END DO
       END DO
       
        do inod=1,4
          k4=(3*inod)
          PhiiV(jelem,k4-2)=coords(1,inod)
          PhiiV(jelem,k4-1)=coords(2,inod)
          PhiiV(jelem,k4)=U(inod)
        end do
   


















       
       
C     ==================================================================
C     ******************************************************************
C     Constructing elemet TYPE 2 (displacement)
C     ******************************************************************
C     ==================================================================
      ELSEIF (JTYPE.EQ.TWO) THEN
      STEPITER=USRVAR(JELEM-N_ELEM,22,1)
      
      DO I=1,8
          USRVAR(JELEM-N_ELEM,24,I)=U(I)
      END DO
      
C     ==================================================================
C     Material parameters
C     ==================================================================
       THCK = PROPS(1)
       PARK = PROPS(2)
       kexp=PROPS(3)
       E1o=PROPS(4) 
       E2o=PROPS(5) 
       G12o=PROPS(6)
       XNU12=PROPS(7)
       Gc1=PROPS(8)
       Gc2=PROPS(9)
       Gc3=PROPS(10)

C     ==================================================================
C     RETRIEVING  RCFALG FROM USERVAR
C     ==================================================================
      rcflag=USRVAR(JELEM-N_ELEM,23,1)
      rav=USRVAR(JELEM-N_ELEM,23,2)
      rbv=USRVAR(JELEM-N_ELEM,23,3)
      NINP=INTDATA(JELEM-N_ELEM,1)
      
C     ==================================================================
C     Initial preparations
C     ==================================================================
       DO K1 = 1, NDOFEL                      
        DO KRHS = 1, NRHS
         RHS(K1,KRHS) = ZERO
        END DO
        DO K2 = 1, NDOFEL
         AMATRX(K2,K1) = ZERO
        END DO
       END DO
C     ==================================================================
C     Local weights
C     ==================================================================
       IF (rcflag.lt.0.5d0) then
         do I=1,4
           AINTW(I)=1.d0
         enddo
       elseif (rcflag.gt.0.5d0) then
         call kinptdata(wght,coord16,NINP)
         do I=1,100
           AINTW(I)=wght(I)
         enddo
       endif

C     ==================================================================
C     Calculating properties at each integration point
C     ==================================================================
       NINPT=NINP*NINP
       DO INPT=1,NINPT
C     Initial variables
        DO I=1,NSTVTT
          SDV(I)=SVARS(NSTVTT*(INPT-1)+I)
        END DO
C
C     Shape functions and local derivatives
        CALL kshapefcn(INPT,delta,rav,rbv,rcflag,AN,dNdxi,dNE,dNdzEXP,
     1   jtype,NINPT)

!       coordinates in global
        xc=COORDS(1,1)*AN(1)+COORDS(1,2)*AN(2)+COORDS(1,3)*AN(3)+
     1     COORDS(1,4)*AN(4)
        yc=COORDS(2,1)*AN(1)+COORDS(2,2)*AN(2)+COORDS(2,3)*AN(3)+
     1     COORDS(2,4)*AN(4)
     
c       call kprofin(props,xc,yc,Es,xnus,Gcs)
       E1x=E1o*exp(Oalpha*yc)
       E2x=E2o*exp(Obeta*yc)
       Gbulk=G12o*exp(Ogamma*yc)
       NU12=XNU12
       NU21=NU12*E2x/E1x  
       GCPAR=Gc2
     
C     Shape functions
        IY=ZERO
        DO I = 1,NNODE
         IX=IY+1
         IY=IX+1
         VNI(1,IX)=AN(I)
         VNI(1,IY)=ZERO
         VNI(2,IX)=ZERO
         VNI(2,IY)=AN(I)
        END DO
C     Jacobian
        DO I = 1,2
         DO J = 1,2
          VJACOB(I,J) = ZERO
          DO K = 1,NNODE
           VJACOB(I,J) = VJACOB(I,J) + COORDS(I,K)*dNdxi(K,J)
          END DO
         END DO
        END DO
C        
        DTM = ZERO
        DTM = VJACOB(1,1)*VJACOB(2,2)-VJACOB(1,2)*VJACOB(2,1)
        IF (DTM.LT.ZERO) THEN
         WRITE(7,*) 'Negative Jacobian',JELEM,dNdxi
         CALL XIT	
        ENDIF
C     Inverse of Jacobian
        VJABOBINV(1,1)=VJACOB(2,2)/DTM
        VJABOBINV(1,2)=-VJACOB(1,2)/DTM
        VJABOBINV(2,1)=-VJACOB(2,1)/DTM
        VJABOBINV(2,2)=VJACOB(1,1)/DTM
C        
C     Derivatives of shape functions respect to global ccordinates
        DO K = 1,NNODE
         DO I = 1,2
          dNdx(K,I) = ZERO
          DO J = 1,2
           dNdx(K,I) = dNdx(K,I) + dNdxi(K,J)*VJABOBINV(J,I)
          END DO
         END DO
        END DO
C
C     Calculating B matrix (B=LN)
       IY=0
       BB=ZERO
       DO INODE=1,NNODE
        IX=IY+1
        IY=IX+1
        BB(1,IX)= dNdx(INODE,1)
        BB(1,IY)= ZERO
        BB(2,IX)= ZERO
        BB(2,IY)= dNdx(INODE,2)
        BB(3,IX)= dNdx(INODE,2)
        BB(3,IY)= dNdx(INODE,1)
       END DO
C
C     ==================================================================
C     Calculating materials stiffness matrix (plane strain)
C     ==================================================================
        DO I=1,3
         DO J=1,3
          CMAT(I,J)=ZERO
         END DO
        END DO
        DO I=1,3
         DO J=1,3
          Tmat(I,J)=ZERO
         END DO
        END DO
        DO I=1,3
         DO J=1,3
          Qmat(I,J)=ZERO
         END DO
        END DO
        Tmat(1,1)=cos(theta)*cos(theta)
        Tmat(1,2)=sin(theta)*sin(theta)
        Tmat(1,3)=2.d0*sin(theta)*cos(theta)
        Tmat(2,1)=sin(theta)*sin(theta)
        Tmat(2,2)=cos(theta)*cos(theta)
        Tmat(2,3)=-2.d0*sin(theta)*cos(theta)
        Tmat(3,1)=-sin(theta)*cos(theta)
        Tmat(3,2)=sin(theta)*cos(theta)
        Tmat(3,3)=cos(theta)*cos(theta)-sin(theta)*sin(theta)

        Rmat(1,1)=1.D0
        Rmat(1,2)=0.D0
        Rmat(1,3)=0.D0
        Rmat(2,1)=0.D0
        Rmat(2,2)=1.D0
        Rmat(2,3)=0.D0
        Rmat(3,1)=0.D0
        Rmat(3,2)=0.D0
        Rmat(3,3)=2.D0

        call inverseJ(Tmat,Tinv) 
        call inverseJ(Rmat,Rinv)

        Qmat(1,1)=E1x/(1.d0-NU12*NU21)
        Qmat(2,2)=E2x/(1.d0-NU12*NU21)
        Qmat(1,2)=NU12*E2x/(1.d0-NU12*NU21)
        Qmat(2,1)=NU12*E2x/(1.d0-NU12*NU21)
        Qmat(3,3)=Gbulk
        
        CMAT=matmul(matmul(matmul(matmul(Tinv,Qmat),Rmat),Tmat),Rinv)

        CCMAT(1,1)=E1x/((ONE+NU12)*(ONE-TWO*NU12))*(ONE-NU12)
        CCMAT(2,2)=E1x/((ONE+NU12)*(ONE-TWO*NU12))*(ONE-NU12)
        CCMAT(3,3)=E1x/((ONE+NU12)*(ONE-TWO*NU12))*(HALF-NU12)
        CCMAT(1,2)=E1x/((ONE+NU12)*(ONE-TWO*NU12))*NU12
        CCMAT(2,1)=E1x/((ONE+NU12)*(ONE-TWO*NU12))*NU12
C
C     ==================================================================
C     Nodal displacements
C     ==================================================================
        DO J=1,2
         ULOC(J)=ZERO
        END DO
        DO J=1,2
         DO I=1,NDOFEL
          ULOC(J)=ULOC(J)+VNI(J,I)*U(I)
         END DO
        END DO  
        DO J=1,2
         SDV(J)=ULOC(J)
        END DO
C   
C     ==================================================================
C     Nodal phase-field
C     ==================================================================
        IF (STEPITER.EQ.ZERO) THEN
         PHASE=USRVAR(JELEM-N_ELEM,17,INPT)
        ELSE
         PHASE=USRVAR(JELEM-N_ELEM,14,INPT)
        ENDIF
C
        
        IF (PHASE.GT.1.D0) THEN
          PHASE=1.D0
        ENDIF
        SDV(14)=PHASE
C     ==================================================================
C     Calculating strain
C     ==================================================================
        DO J=1,3
         EPS(J)=ZERO
        END DO
        DO I=1,3
         DO J=1,NDOFEL
          EPS(I)=EPS(I)+BB(I,J)*U(J)    
         END DO
        END DO
        DO J=1,3
         SDV(J+2)=EPS(J)
        END DO
c
c     ==================================================================
C     Calculating stresses
C     ==================================================================
        DO K1=1,3
         STRESS(k1)=ZERO
        END DO
        DO K1=1,4
         STRESS4(k1)=ZERO
        END DO
        
        call kddsdde(eg,elam,PS,PV,ddsdde)
        DO K1=1,3
         DO K2=1,3
          STRESS(K2)=STRESS(K2)+CMAT(K2,K1)*EPS(K1)
         END DO
        END DO
C$$$$$$         DO K1=1,4
C$$$$$$          DO K2=1,4
C$$$$$$           STRESS4(K2)=STRESS4(K2)+ddsdde(K2,K1)*EPS2(K1)
C$$$$$$          END DO
C$$$$$$         END DO
        DO J=1,3
         SDV(J+5)=STRESS(J)*((ONE-PHASE)**TWO+PARK)
        END DO
        DO J=1,3
         SDV(J+8)=STRESS(J)
        END DO

C$$$$$$         
C$$$$$$          STRESS(1)=STRESS4(1)
C$$$$$$          STRESS(2)=STRESS4(2)
C$$$$$$          STRESS(3)=STRESS4(4)
       
C     ==================================================================
C     Calculating elastic ENERGY
C     ==================================================================  
        psif=(STRESS(1)+abs(STRESS(1)))*(EPS(1)+abs(EPS(1)))/8.d0
        psim1=(STRESS(2)+abs(STRESS(2)))*(EPS(2)+abs(EPS(2)))/8.d0
        psin=(STRESS(1)-abs(STRESS(1)))*(EPS(1)-abs(EPS(1)))/8.d0
        psin1=(STRESS(2)-abs(STRESS(2)))*(EPS(2)-abs(EPS(2)))/8.d0
        psim2=STRESS(3)*EPS(3)
C$$$$$$         
C$$$$$$         psif=STRESS(1)*EPS(1)*HALF
C$$$$$$         psim1=STRESS(2)*EPS(2)*HALF
C$$$$$$         psim2=STRESS(3)*EPS(3)*HALF
                
        ENG=ZERO
        DO K2=1,3
         ENG=ENG+STRESS(K2)*EPS(K2)*HALF
        END DO
C$$$$$$         SDV(12)=ENG*((ONE-PHASE)**TWO+PARK)
        SDV(13)=psif
C$$$$$$         SDV(13)=psif+psim1+psim2
        SDV(15)=psim1
        SDV(16)=psim2
C$$$$$$         ENERGY(2)=psif+psim1+psim2

c        
C     ==================================================================
C     Calculating element stiffness matrix
C     ==================================================================
C
        DO K=1,NDOFEL
         DO L=1,NDOFEL
          DO I=1,3
           DO J=1,3
            AMATRX(K,L)=AMATRX(K,L)+AINTW(INPT)*BB(I,K)*CMAT(I,J)*
     1       BB(J,L)*DTM*THCK*((ONE-PHASE)**TWO+PARK)
           END DO
          END DO
         END DO
        END DO
C       
C     ==================================================================
C     Internal forces (residual vector)
C     ==================================================================
        DO K1=1,NDOFEL
         DO K4=1,3
           RHS(K1,1)=RHS(K1,1)-AINTW(INPT)*BB(K4,K1)*STRESS(K4)*DTM*
     1      THCK*((ONE-PHASE)**TWO+PARK)
         END DO
        END DO
C     
C     ==================================================================
C     Uploading solution dep. variables
C     ==================================================================
        DO I=1,NSTVTT
         SVARS(NSTVTT*(INPT-1)+I)=SDV(I)
         USRVAR(JELEM-N_ELEM,I,INPT)=SVARS(NSTVTT*(INPT-1)+I)
        END DO
       END DO
      ENDIF
C      
      RETURN
      END

      SUBROUTINE SHAPEFUN(AN,dNdxi,xi)
      INCLUDE 'ABA_PARAM.INC'
      Real*8 AN(4),dNdxi(4,2)
      Real*8 XI(2)
      PARAMETER(ZERO=0.D0,ONE=1.D0,MONE=-1.D0,FOUR=4.D0)
C
C     Values of shape functions as a function of local coord.
      AN(1) = ONE/FOUR*(ONE-XI(1))*(ONE-XI(2))
      AN(2) = ONE/FOUR*(ONE+XI(1))*(ONE-XI(2))
      AN(3) = ONE/FOUR*(ONE+XI(1))*(ONE+XI(2))
      AN(4) = ONE/FOUR*(ONE-XI(1))*(ONE+XI(2))
C
C     Derivatives of shape functions respect to local coordinates
      DO I=1,4
        DO J=1,2
            dNdxi(I,J) =  ZERO
        END DO
      END DO
      dNdxi(1,1) =  MONE/FOUR*(ONE-XI(2))
      dNdxi(1,2) =  MONE/FOUR*(ONE-XI(1))
      dNdxi(2,1) =  ONE/FOUR*(ONE-XI(2))
      dNdxi(2,2) =  MONE/FOUR*(ONE+XI(1))
      dNdxi(3,1) =  ONE/FOUR*(ONE+XI(2))
      dNdxi(3,2) =  ONE/FOUR*(ONE+XI(1))
      dNdxi(4,1) =  MONE/FOUR*(ONE+XI(2))
      dNdxi(4,2) =  ONE/FOUR*(ONE-XI(1))
      RETURN
      END

      subroutine kshapefcn(INPT,delta,rav,rbv,rcflag,
     1  AN,dNdxi,dNE,dNdzEXP,jtype,NINPT)
      include 'aba_param.inc'
      
      integer l,INPT,NINPT,NINPTD
      REAL*8 AN(4),dNdxi(4,2)
      REAL*8 dNE(4),dNdzEXP(4,2)
      REAL*8 RNEF(4,2),DNEF(4,2),coord24(2,4)
      REAL*8 RNEB(8,4),DNEB(8,4)
      parameter (gaussCoord=0.577350269d0)
      real *8 g,h,delta(4),dli,rav,rbv,rcflag
      real*8 coord16(100,2),wght(100)
      real*8 dnerror

      data  coord24 /-1.d0, -1.d0,
     2                1.d0, -1.d0,
     3               -1.d0,  1.d0,
     4                1.d0,  1.d0/

      NINPTD=SQRT(NINPT*1.D0)
!     determine (g,h)
      if (rcflag.lt.0.5d0) then
         g=coord24(1,INPT)*gaussCoord
         h=coord24(2,INPT)*gaussCoord
      else
         call kinptdata(wght,coord16,NINPTD)
         IF (rav.lt.1.D0) then
           print*,'RAV & RBV not assigned'
           call xit
         endif  
         g=coord16(INPT,1)
         h=coord16(INPT,2)
      endif
 
!     shape functions 
      AN(1)=(1.d0-g)*(1.d0-h)/4.d0
      AN(2)=(1.d0+g)*(1.d0-h)/4.d0
      AN(3)=(1.d0+g)*(1.d0+h)/4.d0
      AN(4)=(1.d0-g)*(1.d0+h)/4.d0

!     derivative d(Ni)/d(g)
      dNdxi(1,1)=-(1.d0-h)/4.d0
      dNdxi(2,1)=(1.d0-h)/4.d0
      dNdxi(3,1)=(1.d0+h)/4.d0
      dNdxi(4,1)=-(1.d0+h)/4.d0

!     derivative d(Ni)/d(h)
      dNdxi(1,2)=-(1.d0-g)/4.d0
      dNdxi(2,2)=-(1.d0+g)/4.d0
      dNdxi(3,2)=(1.d0+g)/4.d0
      dNdxi(4,2)=(1.d0-g)/4.d0
      
      dli=0.1d0
C$$$$$$       dli=1.1d0
C$$$$$$       if (jtype.eq.1.d0) then
      
      if (rcflag.gt.0.5d0) then
      
      do l=1,4
      
         
            RNEB(1,l)=1-((exp(-dli*(1+g)/4)-1)/(exp(-dli/2)-1))
            RNEB(2,l)=((exp(-dli*(1+g)/4)-1)/(exp(-dli/2)-1))
            RNEB(3,l)=1-((exp(-dli*(1+h)/4)-1)/(exp(-dli/2)-1))
            RNEB(4,l)=((exp(-dli*(1+h)/4)-1)/(exp(-dli/2)-1))
                
            RNEB(5,l)=1-((exp(-dli*(1-g)/4)-1)/(exp(-dli/2)-1))
            RNEB(6,l)=((exp(-dli*(1-g)/4)-1)/(exp(-dli/2)-1))
            RNEB(7,l)=1-((exp(-dli*(1-h)/4)-1)/(exp(-dli/2)-1))
            RNEB(8,l)=((exp(-dli*(1-h)/4)-1)/(exp(-dli/2)-1))

            DNEB(1,l)=(dli/4)*(exp(-dli*(1+g)/4))/(exp(-dli/2)-1)
            DNEB(2,l)=-DNEB(1,l)
            DNEB(3,l)=(dli/4)*(exp(-dli*(1+h)/4))/(exp(-dli/2)-1)
            DNEB(4,l)=-DNEB(3,l)
            
            DNEB(5,l)=(dli/4)*(exp(-dli*(1-g)/4))/(exp(-dli/2)-1)
            DNEB(6,l)=-DNEB(5,l)
            DNEB(7,l)=(dli/4)*(exp(-dli*(1-h)/4))/(exp(-dli/2)-1)
            DNEB(8,l)=-DNEB(7,l)
      end do
      

                    if (rav.eq.1.d0) then
                        RNEF(1,1)=RNEB(1,1)
                        RNEF(1,2)=RNEB(2,1)
                        DNEF(1,1)=DNEB(1,1)
                        DNEF(1,2)=DNEB(2,1)

                        RNEF(3,1)=RNEB(1,3)
                        RNEF(3,2)=RNEB(2,3)
                        DNEF(3,1)=DNEB(1,3)
                        DNEF(3,2)=DNEB(2,3)
                    elseif (rav.eq.2.d0) then
                        RNEF(1,1)=RNEB(6,1)
                        RNEF(1,2)=RNEB(5,1)
                        DNEF(1,1)=DNEB(6,1)
                        DNEF(1,2)=DNEB(5,1)

                        RNEF(3,1)=RNEB(6,3)
                        RNEF(3,2)=RNEB(5,3)
                        DNEF(3,1)=DNEB(6,3)
                        DNEF(3,2)=DNEB(5,3)
                    endif
                    if (rbv.eq.1.d0) then
                        RNEF(2,1)=RNEB(3,2)
                        RNEF(2,2)=RNEB(4,2)
                        DNEF(2,1)=DNEB(3,2)
                        DNEF(2,2)=DNEB(4,2)

                        RNEF(4,1)=RNEB(3,4)
                        RNEF(4,2)=RNEB(4,4)
                        DNEF(4,1)=DNEB(3,4)
                        DNEF(4,2)=DNEB(4,4)                        
                    elseif (rbv.eq.2.d0) then
                        RNEF(2,1)=RNEB(8,2)
                        RNEF(2,2)=RNEB(7,2)
                        DNEF(2,1)=DNEB(8,2)
                        DNEF(2,2)=DNEB(7,2)

                        RNEF(4,1)=RNEB(8,4)
                        RNEF(4,2)=RNEB(7,4)
                        DNEF(4,1)=DNEB(8,4)
                        DNEF(4,2)=DNEB(7,4)
                    endif
      
            dNE(1)=RNEF(1,1)*RNEF(4,1)
            dNE(2)=RNEF(1,2)*RNEF(2,1)
            dNE(3)=RNEF(3,2)*RNEF(2,2)
            dNE(4)=RNEF(3,1)*RNEF(4,2)
            
            dnerror=1-(dNE(1)+dNE(2)+dNE(3)+dNE(4))
            dNE(1)= dNE(1)+(dnerror/4.d0)
            dNE(2)= dNE(2)+(dnerror/4.d0)
            dNE(3)= dNE(3)+(dnerror/4.d0)
            dNE(4)= dNE(4)+(dnerror/4.d0)
            
            dNdzEXP(1, 1) = RNEF(4,1)*DNEF(1,1)
            dNdzEXP(2, 1) = RNEF(2,1)*DNEF(1,2)
            dNdzEXP(3, 1) = RNEF(2,2)*DNEF(3,2)
            dNdzEXP(4, 1) = RNEF(4,2)*DNEF(3,1)
            
            dNdzEXP(1, 2) = RNEF(1,1)*DNEF(4,1)
            dNdzEXP(2, 2) = RNEF(1,2)*DNEF(2,1)
            dNdzEXP(3, 2) = RNEF(3,2)*DNEF(2,2)
            dNdzEXP(4, 2) = RNEF(3,1)*DNEF(4,2)
      elseif (rcflag.lt.0.5d0) then
        dNE=AN
        dNdzEXP=dNdxi
      endif

C$$$$$$       elseif (jtype.eq.2.d0) then
C$$$$$$         dNE=AN
C$$$$$$         dNdzEXP=dNdxi
C$$$$$$       endif
      

!     Manual Override to linear shape functions !!!!!!!!!!!!!!!!!!!!!!!!!!      
C$$$$$$       dNE=AN
C$$$$$$       dNdzEXP=dNdxi
!     Manual Override to linear shape functions !!!!!!!!!!!!!!!!!!!!!!!!!!
        

      
      return
      end
      
c     *****************************************************************       
       subroutine kprofin(props,xc,yc,Es,xnus,Gcs)
       include 'aba_param.inc' !implicit real(a-h o-z)
       dimension stran(4),PS(3),AN(3,3),props(*)
       
       real *8 xc,yc,Es,xnus,E1,E2,xnu1,xnu2,Gc1,Gc2,ynorm,xnorm,v1,v2,
     1 k1,k2,g1,g2,kexp,ke,Gcs
     
       kexp=PROPS(3)
       E1=PROPS(4) 
       E2=PROPS(5) 
       XNU1=PROPS(6)
       XNU2=PROPS(7)
       Gc1=PROPS(8)
       Gc2=PROPS(9)

       ynorm=(yc/17.D0)-0.5d0
       
       v1=(0.5d0-ynorm)**kexp
       v2=1.d0-v1

       Es=E1+(E2-E1)*v1
       xnus=XNU1+(XNU2-XNU1)*v1
       Gcs=Gc1+(Gc2-Gc1)*v1

       return 
       end  
       
c***********************************************************************
      subroutine kddsdde(eg,elam,PS,PV,ddsdde)

      include 'aba_param.inc' !implicit real(a-h o-z)

      dimension kkddsdde(6,6),PS(3),ddsdde(4,4),PV(3,3)
      real*8 eg,elam
      ddsdde=0.d0
      kkddsdde=0.d0
      
!     trace of strain
      TRP=PS(1)+PS(2)+PS(3)
      if (TRP.ge.0.d0) then
       kkddsdde(1:3,1:3)=elam
      else
       kkddsdde(1:3,1:3)=elam
      endif
      do ki=1,3
       do kj=1,3
        do kk=1,3
         do kl=1,3
          if (ki.le.kj .and. kk.le.kl) then
           if (ki.eq.kj) then
            k1=ki
           else
            select case(10*ki+kj)
            case (12)
             k1=4
            case (23)
             k1=6
            case (13)
             k1=5
            end select
           endif
           if (kk.eq.kl) then
            k2=kk
           else
            select case(10*kk+kl)
            case (12)
             k2=4
            case (23)
             k2=6
            case (13)
             k2=5
            end select
           endif

           xKPp=0.d0
           xKPn=0.d0
           do ka=1,3
            do kb=1,3
             if (ka.eq.kb) then
              if (PS(ka) .gt. 0.d0) then
               kH=1.d0
              else
               kH=0.d0
              endif
              xkNP1=PV(ka,ki)*PV(ka,kj)*PV(kb,kk)*PV(kb,kl)
              xKPp=xKPp+kH*xkNP1
              xKPn=xKPn+(1.d0-kH)*xkNP1
             else
              xkNP1=PV(ka,ki)*PV(kb,kj)*(PV(ka,kk)*PV(kb,kl)+PV(kb,
     1        kk)*PV(ka,kl))
              if (PS(ka)==PS(kb)) then
               if(PS(ka).gt.0.d0) then
                xKPp=xKPp+5.d-1*xkNP1
               else
                xKPn=xKPn+5.d-1*xkNP1
               endif
              else
               xKPp=xKPp+5.d-1*(((PS(ka)+abs(PS(ka)))/2-(PS(kb)+
     1         abs(PS(kb)))/2)/(PS(ka)-PS(kb)))*xkNP1

               xKPn=xKPn+5.d-1*(((PS(ka)-abs(PS(ka)))/2-(PS(kb)-
     1         abs(PS(kb)))/2)/(PS(ka)-PS(kb)))*xkNP1
              endif

             endif
            end do
           end do
           kkddsdde(k1,k2)=kkddsdde(k1,k2)+2.d0*eg*(xKPp+xKPn)
          endif
         end do
        end do
       end do
      end do

      ddsdde=kkddsdde(1:ntens,1:ntens)
      return
      end

c     *****************************************************************       
      subroutine kinptdata(wght,coord16,NINPT)
      integer k1,k2,kindex,ninpt,ninptloc
      real *8 coord16(100,2), wght(100)
      dimension gp(10),gw(10)
      
      ninptloc=NINPT
      if (ninptloc.eq.2) then
       gp(1) = -0.57735027
       gp(2) = 0.57735027
       gw(1) = 1.00000000
       gw(2) = 1.00000000
      elseif (ninptloc.eq.3) then
       gp(1) = 0.00000000
       gp(2) = -0.77459667
       gp(3) = 0.77459667
       gw(1) = 0.88888888
       gw(2) = 0.55555556
       gw(3) = 0.55555556
      elseif (ninptloc.eq.4) then
       gp(1) = -0.33998104
       gp(2) = 0.33998104
       gp(3) = -0.86113631
       gp(4) = 0.86113631
       gw(1) = 0.65214515
       gw(2) = 0.65214515 
       gw(3) = 0.34785484
       gw(4) = 0.34785484
      elseif (ninptloc.eq.5) then
       gp(1) = 0.0000000
       gp(2) = -0.53846931
       gp(3) = 0.53846931
       gp(4) = -0.90617984
       gp(5) = 0.90617984
       gw(1) = 0.56888888
       gw(2) = 0.47862867
       gw(3) = 0.47862867
       gw(4) = 0.23692688
       gw(5) = 0.23692688
      elseif (ninptloc.eq.6) then
       gp(1) = 0.66120938
       gp(2) = -0.66120938
       gp(3) = -0.2386191
       gp(4) = 0.2386191
       gp(5) = -0.93246951
       gp(6) = 0.93246951
       gw(1) = 0.36076157
       gw(2) = 0.36076157
       gw(3) = 0.46791393
       gw(4) = 0.46791393
       gw(5) = 0.17132449
       gw(6) = 0.17132449
      elseif (ninptloc.eq.7) then
       gp(1) = 0.00000000
       gp(2) = 0.40584515
       gp(3) = -0.40584515
       gp(4) = -0.7415311
       gp(5) = 0.7415311
       gp(6) = -0.9491079
       gp(7) = 0.9491079
       gw(1) = 0.41795918
       gw(2) = 0.38183005
       gw(3) = 0.38183005
       gw(4) = 0.27970539
       gw(5) = 0.27970539
       gw(6) = 0.12948496
       gw(7) = 0.12948496
      elseif (ninptloc.eq.8) then
       gp(1) = -0.18343464
       gp(2) = 0.18343464
       gp(3) = -0.52553240
       gp(4) = 0.52553240
       gp(5) = -0.79666647
       gp(6) = 0.79666647
       gp(7) = -0.96028985
       gp(8) = 0.96028985
       gw(1) = 0.36268378
       gw(2) = 0.36268378
       gw(3) = 0.31370664
       gw(4) = 0.31370664
       gw(5) = 0.22238103
       gw(6) = 0.22238103
       gw(7) = 0.10122853
       gw(8) = 0.10122853
      elseif (ninptloc.eq.9) then
       gp(1) = 0.000000000
       gp(2) = -0.83603110
       gp(3) = 0.83603110
       gp(4) = -0.96816023
       gp(5) = 0.96816023
       gp(6) = -0.32425342
       gp(7) = 0.32425342
       gp(8) = -0.61337143
       gp(9) = 0.61337143
       gw(1) = 0.33023935
       gw(2) = 0.18064816
       gw(3) = 0.18064816
       gw(4) = 0.08127438
       gw(5) = 0.08127438
       gw(6) = 0.31234707
       gw(7) = 0.31234707
       gw(8) = 0.26061069
       gw(9) = 0.26061069
      elseif (ninptloc.eq.10) then
       gp(1) = -0.97390653
       gp(2) = -0.86506337
       gp(3) = -0.67940957
       gp(4) = -0.43339539
       gp(5) = -0.14887434
       gp(6) = 0.14887434
       gp(7) = 0.43339539
       gp(8) = 0.67940957
       gp(9) = 0.86506337
       gp(10) = 0.97390653
       gw(1) = 0.06667134
       gw(2) = 0.14945135
       gw(3) = 0.21908636
       gw(4) = 0.26926672
       gw(5) = 0.29552422
       gw(6) = 0.29552422
       gw(7) = 0.26926672
       gw(8) = 0.21908636
       gw(9) = 0.14945135
       gw(10) = 0.06667134
      endif
       do k1=1,ninptloc
        do k2=1,ninptloc
          kindex=(k1-1)*ninptloc+k2
          wght(kindex)=gw(k1)*gw(k2)
          coord16(kindex,1)=gp(k1)
          coord16(kindex,2)=gp(k2)
        end do
       end do
      
      return
      end
      
c     *****************************************************************
      subroutine inverseJ(bM,bMI)
  
!      include 'vaba_param.inc'
      
      parameter(zero=0.d0,one=1.d0,eight=8.d0)
  
      dimension bM(3,3),bMI(3,3)
          
      !local variables
      dimension AdjM(3,3)
    
      AdjM(1:3,1:3) = zero
      
      AdjM(1,1) =  bM(2,2)*bM(3,3)-bM(2,3)*bM(3,2)
      AdjM(2,1) = -bM(2,1)*bM(3,3)+bM(2,3)*bM(3,1)
      AdjM(3,1) =  bM(2,1)*bM(3,2)-bM(2,2)*bM(3,1)
    
      AdjM(1,2) = -bM(1,2)*bM(3,3)+bM(1,3)*bM(3,2)
      AdjM(2,2) =  bM(1,1)*bM(3,3)-bM(1,3)*bM(3,1)
      AdjM(3,2) = -bM(1,1)*bM(3,2)+bM(1,2)*bM(3,1)
    
      AdjM(1,3) =  bM(1,2)*bM(2,3)-bM(2,2)*bM(1,3)
      AdjM(2,3) = -bM(1,1)*bM(2,3)+bM(2,1)*bM(1,3)
      AdjM(3,3) =  bM(1,1)*bM(2,2)-bM(2,1)*bM(1,2)
    
      
      DJ = bM(1,1)*AdjM(1,1)+bM(1,2)*AdjM(2,1)+bM(1,3)*AdjM(3,1)

      if(DJ==0) then
        write(*,*) "warning: zero Jacob value!"
      else
        bMI(:,:) = AdjM(:,:)/DJ
      end if

      return
      end subroutine
c     ***************************************************************** 
      SUBROUTINE KORIENT(U,PHISWITCH,rav2,rbv2,rcflag2,jelem,mrkr,NINP)
      real*8 U(4)
      INTEGER JELEM,NINP
      REAL*8 rcflag2,rav2,rbv2,SPHI,PHISWITCH
      real*8 dpa,dpb,mrkr(1,3)
      
      SPHI=0.D0
      SPHI=U(1)+U(2)+U(3)+U(4)
      PHISWITCH=1.0d0
C$$$$$$       if (sphi.lt.0.d0) sphi=0.d0 then
C$$$$$$       rav2=0.d0
C$$$$$$       rbv2=0.d0
C$$$$$$       endif
      
C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,mrkr(1,1),mrkr(1,2),mrkr(1,3)
C$$$$$$         endif



      IF (SPHI.gt.PHISWITCH) THEN
        rcflag2=1.D0
        

        NINP=2
        if (SPHI.GE.3.5d0) then
          NINP=4
        else if (SPHI.GE.3.0d0) then
          NINP=4
        else if (SPHI.GE.2.5d0) then
          NINP=4
        else if (SPHI.GE.2.0d0) then
          NINP=4
        else if (SPHI.GE.1.5d0) then
          NINP=4
        else if (SPHI.GE.1.0d0) then
          NINP=4
        else if (SPHI.GE.0.5d0) then
          NINP=4
        endif  
        
        if (abs(U(1)-U(2)).ge.abs(U(4)-U(3))) then
          if (U(1).ge.U(2)) then
            dpa=1.d0
          else
            dpa=2.d0
          endif
        else
          if (U(4).ge.U(3)) then
            dpa=1.d0
          else
            dpa=2.d0
          endif
        endif
        if (abs(U(2)-U(3)).ge.abs(U(1)-U(4))) then
          if (U(2).ge.U(3)) then
            dpb=1.d0
          else
            dpb=2.d0
          endif
        else
          if (U(1).ge.U(4)) then
            dpb=1.d0
          else
            dpb=2.d0
          endif
        endif
        if (dpa.eq.0.d0) then 
          print*,'ERROR DPA NOT ASSIGNED'
        endif
         if (dpb.eq.0.d0) then 
          print*,'ERROR DPB NOT ASSIGNED'
         endif
         rav2=dpa
         rbv2=dpb
      ELSE
         rcflag2=0.D0
         NINP=2
         rav2=0.D0
         rbv2=0.D0
      ENDIF
      
C$$$$$$         if (jelem.eq.102.d0) then
C$$$$$$           print*,rcflag2,rav2,rbv2
C$$$$$$         endif

      return
      end
       
c     *****************************************************************
C      
C Subroutine UMAT  : 
C Dummy material
C
C ==============================================================
C !!! NOTE: N_ELEM has to be changed according to the UEL !!!!!
C ==============================================================
C
       SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,
     2 TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,MATERL,NDI,NSHR,NTENS,
     3 NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,CELENT,
     4 DFGRD0,DFGRD1,NOEL,NPT,KSLAY,KSPT,KSTEP,KINC)
C
      use kvisual
      INCLUDE 'ABA_PARAM.INC'
C
       CHARACTER*80 CMNAME
       DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1 DDSDDE(NTENS,NTENS),
     2 DDSDDT(NTENS),DRPLDE(NTENS),
     3 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     4 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3)
C 
       PARAMETER (ONE=1.0,TWO=2.0,THREE=3.0,SIX=6.0, HALF =0.5,
     1 NSTV=24) 
       DATA NEWTON,TOLER/40,1.D-10/ 
C       
       COMMON/KUSER/USRVAR(N_ELEM,NSTV,100)
C 
C ----------------------------------------------------------- 
C          Material properties
C ----------------------------------------------------------- 
C          PROPS(1) - Young's modulus 
C          PROPS(2) - Poisson ratio 
C ----------------------------------------------------------- 
C
C	Elastic properties
C
       EMOD=PROPS(1)
       ENU=PROPS(2)
       EG=EMOD/(TWO*(ONE+ENU))
       EG2=EG*TWO
       ELAM=EG2*ENU/(ONE-TWO*ENU)
C
C	Stiffness tensor
C
       DO K1=1, NTENS
        DO K2=1, NTENS
         DDSDDE(K2, K1)=0.0
        END DO
       END DO
C
       DO K1=1, NDI
        DO K2=1, NDI
         DDSDDE(K2, K1)=ELAM
        END DO
        DDSDDE(K1, K1)=EG2+ELAM
       END DO 
C
       DO K1=NDI+1, NTENS
        DDSDDE(K1, K1)=EG
       END DO
C
C	Calculate Stresses
C
       DO K1=1, NTENS
        DO K2=1, NTENS
         STRESS(K2)=STRESS(K2)+DDSDDE(K2, K1)*DSTRAN(K1)
        END DO
       END DO 
C
       NELEMAN=NOEL-TWO*N_ELEM

C       
       DO I=1,NSTATV
        STATEV(I)=USRVAR(NELEMAN,I,NPT)
       END DO
C       
       RETURN
       END      
      