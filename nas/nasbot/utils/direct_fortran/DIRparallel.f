C+-----------------------------------------------------------------------+
C| Program       : Direct.f (subfile DIRseriell.f)                       |
C| Last modified : 02-22-01                                              |
C| Written by    : Joerg Gablonsky                                       |
C| Subroutines, which differ depending on the serial or parallel version.|
C+-----------------------------------------------------------------------+

C+-----------------------------------------------------------------------+
C| Parallel Direct. This routine replaces the normal main routine DIRect.|
C| In it, we find out if this pe is the master or slave. If it is the    |
C| master, it calls the serial DIRect main routine. The only routine that|
C| has to change for parallel Direct is DIRSamplef, where the actual     |
C| sampling of the function is done. If we are on the slave, wait for    |
C| either the coordinates of a point to sample the function or the       |
C| termination signal.                                                   |
C+-----------------------------------------------------------------------+
      SUBROUTINE ParDirect(fcn, x, n, eps, maxf, maxT, fmin, l, u,
     +                  algmethod, Ierror, logfile, 
     +                  fglobal, fglper, volper, sigmaper,
     +                  iidata, iisize, ddata, idsize, cdata, icsize)


      IMPLICIT None
C+-----------------------------------------------------------------------+
C| Parameters                                                            |
C+-----------------------------------------------------------------------+

C+-----------------------------------------------------------------------+
C| The maximum of function evaluations allowed.                          |
C| The maximum dept of the algorithm.                                    |
C| The maximum number of divisions allowed.                              |
C| The maximal dimension of the problem.                                 |
C+-----------------------------------------------------------------------+
      INTEGER maxfunc, maxdeep, maxdiv, MaxDim, mdeep
      PARAMETER (Maxfunc = 90000)
      PARAMETER (maxdeep = 600)
      PARAMETER (maxdiv = 3000)
      PARAMETER (MaxDim = 64)

C+-----------------------------------------------------------------------+
C| Global Variables.                                                     |
C+-----------------------------------------------------------------------+
      Integer JONES
      COMMON /directcontrol/ JONES

C+-----------------------------------------------------------------------+
C| External Variables.                                                   |
C+-----------------------------------------------------------------------+
      EXTERNAL fcn
      Integer n, maxf, maxT, algmethod, Ierror, logfile, dwrit
      Double Precision  x(n),fmin,eps,l(n),u(n)
      Double Precision fglobal, fglper, volper, sigmaper, fscale

C+-----------------------------------------------------------------------+
C| User Variables.                                                       |
C| These can be used to pass user defined data to the function to be     |
C| optimized.                                                            |
C+-----------------------------------------------------------------------+
      INTEGER iisize, idsize, icsize
      INTEGER iidata(iisize)
      Double Precision ddata(idsize)
      Character*40 cdata(icsize)
C+-----------------------------------------------------------------------+
C| Parallel programming variables                                        |
C+-----------------------------------------------------------------------+
        integer mytid, tid, nprocs, istat
        integer maxprocs
C       maxprocs should be >= the number of processes used for DIRECT
        parameter(maxprocs = 360)
        integer tids(maxprocs)
        integer flag, kret
        Real*8 fval
C+-----------------------------------------------------------------------+
C| End of parallel programming variables                                 |
C+-----------------------------------------------------------------------+
C+-----------------------------------------------------------------------+
C| Internal variables                                                    |
C+-----------------------------------------------------------------------+
        Integer i,k

C+-----------------------------------------------------------------------+
C| JG 02/28/01 Begin of parallel additions                               |
C| DETERMINE MASTER PROCESSOR. GET TIDS OF ALL PROCESSORS.               |
C+-----------------------------------------------------------------------+
      call getmytidIF(mytid)
      call getnprocsIF(nprocs)
      call gettidIF(0, tids(1))
C+-----------------------------------------------------------------------+
C| If I am the master get the other tids and start running DIRECT.       |
C| Otherwise, branch off to do function evaluations.                     |
C+-----------------------------------------------------------------------+
      if (mytid.eq.tids(1)) then
        do 46 i = 1, nprocs-1
              call gettidIF(i, tids(i+1))
 46     continue
C+-----------------------------------------------------------------------+
C| Call Direct main routine. This routine calls DIRSamplef for the       |
C| function evaluations, which are then done in parallel.                |
C+-----------------------------------------------------------------------+
        Call Direct(fcn, x, n, eps, maxf, maxT, fmin, l, u,
     +                  algmethod, Ierror, logfile, 
     +                  fglobal, fglper, volper, sigmaper,
     +                  iidata, iisize, ddata, idsize, cdata, icsize)

C+-----------------------------------------------------------------------+
C| Send exit message to rest of pe's.                                    |
C+-----------------------------------------------------------------------+
        flag = 0
        do 200, tid=2,nprocs
          call mastersendif(tids(tid), tids(tid), n, flag,
     +         flag, x(1), u, l, x)
 200    continue

      else
C+-----------------------------------------------------------------------+
C| This is what the slaves do!!                                          |
C+-----------------------------------------------------------------------+
C+-----------------------------------------------------------------------+
C|   Receive the first point from the master processor.                  |
C+-----------------------------------------------------------------------+
        CALL slaverecvIF(tids(1), -1, n, flag, k, fscale, u, l, x)
C+-----------------------------------------------------------------------+
C| Repeat until master signals to stop.                                  |
C+-----------------------------------------------------------------------+
        do while(flag.gt.0)
C+-----------------------------------------------------------------------+
C| Evaluate f(x).                                                        |
C+-----------------------------------------------------------------------+
          CALL DIRinfcn(fcn,x,l,u,n,fval,kret,
     +                 iidata, iisize, ddata, idsize, cdata, icsize)
C+-----------------------------------------------------------------------+
C| Send result and wait for next point / message with signal to stop.    |
C+-----------------------------------------------------------------------+
          CALL slavesendIF(tids(1), mytid, k, mytid, fval, kret)
          CALL slaverecvIF(tids(1), -1, n, flag, k, fscale, u, l, x)
        end do
      end if


      end
C+-----------------------------------------------------------------------+
C| Subroutine for sampling. This sampling is done in parallel, the master|
C| prozessor is also evaluating the function sometimes.                  |
C+-----------------------------------------------------------------------+
      SUBROUTINE DIRSamplef(c,ArrayI,delta,sample,new,length,
     +           dwrit,logfile,f,free,maxI,point,fcn,x,l,fmin,
     +           minpos,u,n,maxfunc,maxdeep,oops,fmax,
     +           Ifeasiblef,IInfesiblef,
     +           iidata, iisize, ddata, idsize, cdata, icsize) 
      IMPLICIT None

C+-----------------------------------------------------------------------+
C| JG 07/16/01 fcn must be declared external.                            |
C+-----------------------------------------------------------------------+
      EXTERNAL fcn

      INTEGER n,maxfunc,maxdeep,oops
      INTEGER maxI,ArrayI(n),sample,new
      INTEGER length(maxfunc,n),free,point(maxfunc),i
C+-----------------------------------------------------------------------+
C| JG 07/16/01 Removed fcn.                                              |
C+-----------------------------------------------------------------------+
      DOUBLE PRECISION c(maxfunc,n),delta,x(n)
      Double Precision l(n),u(n),f(maxfunc,2)
      Double Precision fmin, fhelp
      INTEGER pos,j,dwrit,logfile,minpos
      INTEGER helppoint,kret, DIRgetmaxDeep, oldpos
C+-----------------------------------------------------------------------+
C| JG 01/22/01 Added variable to keep track of the maximum value found.  |
C|             Added variable to keep track if feasible point was found. |
C+-----------------------------------------------------------------------+
      Double Precision fmax
      Integer Ifeasiblef,IInfesiblef
C+-----------------------------------------------------------------------+
C| Variables to pass user defined data to the function to be optimized.  |
C+-----------------------------------------------------------------------+
      INTEGER iisize, idsize, icsize
      INTEGER iidata(iisize)
      Double Precision ddata(idsize)
      Character*40 cdata(icsize)

C+-----------------------------------------------------------------------+
C| Parallel programming variables.                                       |
C+-----------------------------------------------------------------------+
      integer nprocs, maxprocs , mytid
C JG 09/05/00 Increase this if more processors are used.
      parameter (maxprocs=360)
      integer tids(maxprocs)
      integer k, tid, flag, flag2, istat, datarec
      Integer npts

C+-----------------------------------------------------------------------+
C| Find out the id's of all processors.                                  |
C+-----------------------------------------------------------------------+
      call getnprocsIF(nprocs)
      do 46 i = 0, nprocs-1
         call gettidIF(i, tids(i+1))
46    continue
      

C+-----------------------------------------------------------------------+
C| Set the pointer to the first function to be evaluated,                |
C| store this position also in helppoint.                                |
C+-----------------------------------------------------------------------+
      pos = new
      helppoint = pos
C+-----------------------------------------------------------------------+
C| Iterate over all points, where the function should be                 |
C| evaluated.                                                            |
C+-----------------------------------------------------------------------+
      flag = 1
      npts = maxI + maxI
      k = 1
      do while(k.le.npts.and.k.lt.nprocs)
C+-----------------------------------------------------------------------+
C| tid is the id of the prozessor the next points is send to.            |
C+-----------------------------------------------------------------------+
         tid = k+1
C+-----------------------------------------------------------------------+
C| Copy the position into the helparray x.                               |
C+-----------------------------------------------------------------------+
         DO 60,i=1,n
           x(i) = c(pos,i)
60       CONTINUE
C+-----------------------------------------------------------------------+
C| Send the point.                                                       |
C+-----------------------------------------------------------------------+
         call mastersendIF(tids(tid),tids(tid), n, flag, pos,
     +        x(1), u, l, x)
         k = k + 1
         pos = point(pos)
C+-----------------------------------------------------------------------+
C| Get the next point.                                                   |
C+-----------------------------------------------------------------------+
      end do
        

C+-----------------------------------------------------------------------+
C|  Get data until it is all received.                                   |
C+-----------------------------------------------------------------------+
        datarec = 0

        do while (datarec.lt.npts)          
           if(((dble(datarec)/dble(nprocs)-datarec/nprocs).lt.1D-5)
     *        .and.k.le.npts) then
             DO 165,i=1,n
               x(i) = c(pos,i)
165          CONTINUE
             CALL DIRinfcn(fcn,x,l,u,n,fhelp,kret,
     +                 iidata, iisize, ddata, idsize, cdata, icsize)
             oldpos = pos 
             f(oldpos,1) = fhelp
             datarec = datarec + 1
C+-----------------------------------------------------------------------+
C| Remember if an infeasible point has been found.                       |
C+-----------------------------------------------------------------------+
             IInfesiblef = max(IInfesiblef,kret)
             if (kret .eq. 0) then
C+-----------------------------------------------------------------------+
C| if the function evaluation was O.K., set the flag in                  |
C| f(pos,2).                                                             |
C+-----------------------------------------------------------------------+
               f(oldpos,2) = 0.D0
               Ifeasiblef = 0
C+-----------------------------------------------------------------------+
C| JG 01/22/01 Added variable to keep track of the maximum value found.  |
C+-----------------------------------------------------------------------+
               fmax = max(f(pos,1),fmax)
             end if
C+-----------------------------------------------------------------------+
C| Remember if an infeasible point has been found.                       |
C+-----------------------------------------------------------------------+
             IInfesiblef = max(IInfesiblef,kret)
             if (kret .eq. 1) then
C+-----------------------------------------------------------------------+
C| If the function could not be evaluated at the given point,            |
C| set flag to mark this (f(pos,2) and store the maximum                 |
C| box-sidelength in f(pos,1).                                           |
C+-----------------------------------------------------------------------+
               f(oldpos,2) = 2.D0
               f(oldpos,1) = fmax
             end if
C+-----------------------------------------------------------------------+
C| If the function could not be evaluated due to a failure in            |
C| the setup, mark this.                                                 |
C+-----------------------------------------------------------------------+
             if (kret .eq. -1) then
                f(oldpos,2) = -1.D0
             end if
             k = k + 1
             pos = point(pos)
           end if
C+-----------------------------------------------------------------------+
C| Recover where to store the value.                                     |
C+-----------------------------------------------------------------------+
          call masterrecvIF(-1, -1, oldpos, tid, fhelp, kret)
          f(oldpos,1) = fhelp
          datarec = datarec + 1
C+-----------------------------------------------------------------------+
C| Remember if an infeasible point has been found.                       |
C+-----------------------------------------------------------------------+
          IInfesiblef = max(IInfesiblef,kret)

          if (kret .eq. 0) then
C+-----------------------------------------------------------------------+
C| if the function evaluation was O.K., set the flag in                  |
C| f(pos,2).                                                             |
C+-----------------------------------------------------------------------+
             f(oldpos,2) = 0.D0
             Ifeasiblef = 0
C+-----------------------------------------------------------------------+
C| JG 01/22/01 Added variable to keep track of the maximum value found.  |
C+-----------------------------------------------------------------------+
             fmax = max(f(oldpos,1),fmax)
          end if
          if (kret .eq. 1) then
C+-----------------------------------------------------------------------+
C| If the function could not be evaluated at the given point,            |
C| set flag to mark this (f(pos,2) and store the maximum                 |
C| box-sidelength in f(pos,1).                                           |
C+-----------------------------------------------------------------------+
             f(oldpos,2) = 2.D0
             f(oldpos,1) = fmax
          end if
C+-----------------------------------------------------------------------+
C| If the function could not be evaluated due to a failure in            |
C| the setup, mark this.                                                 |
C+-----------------------------------------------------------------------+
          if (kret .eq. -1) then
            f(oldpos,2) = -1.D0
          end if


C+-----------------------------------------------------------------------+
C|         Send data until it is all sent.                               |
C+-----------------------------------------------------------------------+
          if (k.le.npts) then
C+-----------------------------------------------------------------------+
C| Copy the position into the helparray x.                               |
C+-----------------------------------------------------------------------+
            DO 160,i=1,n
               x(i) = c(pos,i)
160         CONTINUE
            call mastersendIF(tid,tid, n, flag, pos,
     +        x(1), u, l, x)
            k = k + 1
            pos = point(pos)
          end if
        end do
      pos = helppoint
C+-----------------------------------------------------------------------+
C| Iterate over all evaluated points and see, if the minimal             |
C| value of the function has changed. If this has happend,               |
C| store the minimal value and its position in the array.                |
C| Attention: Only valied values are checked!!                           |
C+-----------------------------------------------------------------------+
      DO 50,j=1,maxI + maxI
        IF ((f(pos,1) .LT. fmin) .and. (f(pos,2) .eq. 0)) THEN
          fmin = f(pos,1) 
          minpos = pos
        END IF
        pos = point(pos)
50    CONTINUE
      END

C+-----------------------------------------------------------------------+
C| Problem-specific Initialisation                                       |
C+-----------------------------------------------------------------------+
      SUBROUTINE DIRInitSpecific(x,n)
      IMPLICIT None
      Integer n
      Double Precision x(n)
C+-----------------------------------------------------------------------+
C| Problem - specific variables !                                        |
C+-----------------------------------------------------------------------+

C+-----------------------------------------------------------------------+
C| End of problem - specific variables !                                 |
C+-----------------------------------------------------------------------+

C+-----------------------------------------------------------------------+
C| Start of problem-specific initialisation                              |
C+-----------------------------------------------------------------------+

C+-----------------------------------------------------------------------+
C| End of problem-specific initialisation                                |
C+-----------------------------------------------------------------------+
      end
