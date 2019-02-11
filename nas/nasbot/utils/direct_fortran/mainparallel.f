C+-----------------------------------------------------------------------+
C| Program       : mainparallel.f                                        |
C| Last modified : 07-16-2001                                            |
C| Written by    : Owen Esslinger, Joerg Gablonsky, Alton Patrick        |
C| The main program to run DIRECT on test problems in parallel.          |
C+-----------------------------------------------------------------------+
      PROGRAM main
	
      IMPLICIT NONE
      INTEGER Maxdim
      PARAMETER (Maxdim = 128)      
      integer i, problem
      external myfunc
      
C+----------------------------------------------------------------------+
C| DIRECT specific variables.                                           |
C+----------------------------------------------------------------------+
      Double Precision DIReps, DIRf
      Integer DIRmaxf, DIRmaxT
      Integer DIRalg
      Integer IError, logfile
      Double Precision fglobal, fglper, volper, sigmaper
      double precision u(Maxdim), l(Maxdim)
      integer n
      Double Precision DIRx(Maxdim)

C+-----------------------------------------------------------------------+
C| Variables to pass user defined data to the function to be optimized.  |
C+-----------------------------------------------------------------------+
      INTEGER iisize, idsize, icsize
      Parameter (iisize = 300)
      Parameter (idsize = 300)
      Parameter (icsize = 30)
      INTEGER iidata(iisize)
      Double Precision ddata(idsize)
      Character*40 cdata(icsize)
      
      Integer resultfile
      double precision per2
      real dtime,diff,tarray(2), T1, T2
C+----------------------------------------------------------------------+
C| Parallel programming specific variables.                             |
C+----------------------------------------------------------------------+
      integer maxprocs, nprocs
C     maxprocs should be >= the number of processes used for DIRECT
      parameter(maxprocs = 360)
      integer mytid, tids(maxprocs)
      
C+----------------------------------------------------------------------+
C| Initialize parallel computing.                                       |
C+----------------------------------------------------------------------+
      CALL comminitIF()
C+----------------------------------------------------------------------+
C| Need nprocs to call commexitIF at the end.                           |
C+----------------------------------------------------------------------+
      call getnprocsIF(nprocs)
C+----------------------------------------------------------------------+
C| DETERMINE MASTER PROCESSOR. GET TIDS OF ALL PROCESSORS.              |
C+----------------------------------------------------------------------+
      call getmytidIF(mytid)
      call gettidIF(0, tids(1))
C+----------------------------------------------------------------------+
C| If I am the master, open output and resultfile.                      |
C+----------------------------------------------------------------------+
      if (mytid.eq.tids(1)) then

C+----------------------------------------------------------------------+
C|  Define and open the logfile and the resultfile, where we store the  |
C|  results of the run. We save the problem number, the number of       |
C|  function evaluations needed, the time used by DIRECT, the percent   |
C|  error and a flag signaling if we used DIRECT or DIRECT-l in the     |
C|  resultfile.                                                         |
C+----------------------------------------------------------------------+
        logfile    = 2
        resultfile = 29
        open(logfile, file='direct.out')
        open(resultfile, file='result.out')
        CALL mainheader(logfile)
      else
        logfile = 0
      end if

C+----------------------------------------------------------------------+
C| Read in the problem specific data and the parameters for DIRECT.     |
C+----------------------------------------------------------------------+
      CALL inputdata(n, u, l, logfile, DIReps, DIRmaxf,
     +       DIRmaxT, problem, Maxdim, DIRalg, fglobal, 
     +       fglper, volper, sigmaper, 
     +       iidata, iisize, ddata, idsize, cdata, icsize)
C+----------------------------------------------------------------------+
C| Initialize and start the timing.                                     |
C+----------------------------------------------------------------------+
C        tarray(1) = 0.D0
C        tarray(2) = 0.D0
C        diff=dtime(tarray)
C SP/2 Timing
        CALL CPU_TIME(T1)
C+----------------------------------------------------------------------+
C| If the budget is lower zero, multiply the absolute value of the      |
C| budget by the dimension of the problem.                              |
C+----------------------------------------------------------------------+
        if (DIRmaxf .lt. 0) then
           DIRmaxf = -DIRmaxf*n
        endif
 
C+----------------------------------------------------------------------+
C|  For some problems, we need to store some specific data in the user  |
C|  variables.                                                          |
C+----------------------------------------------------------------------+
        if ((problem .ge. 5) .and. (problem .LE. 7)) then
           iidata(1) = 5
           if (problem .EQ. 6) THEN
              iidata(1) = 7
           else if (problem .EQ. 7) THEN
            iidata(1) = 10
           END IF
         else if ((problem .ge. 8) .and. (problem .LE. 9)) then
           n = 3
           iidata(1) = 4
           if (problem .EQ. 9) THEN
              iidata(1) = 4
              n = 6
           END IF
         end if
C+----------------------------------------------------------------------+
C| Call the optimization method.                                        |
C+----------------------------------------------------------------------+
        CALL ParDIRect(myfunc, DIRx, n, DIReps, DIRmaxf, DIRmaxT,
     +              DIRf, l, u, DIRalg, Ierror, logfile, 
     +              fglobal, fglper, volper, sigmaper,
     +              iidata, iisize, ddata, idsize, cdata, icsize)
C+----------------------------------------------------------------------+
C| If I am the master, give out the results of the optimization.        |
C+----------------------------------------------------------------------+
      if (mytid.eq.tids(1)) then
C+----------------------------------------------------------------------+
C| Give out the results of the optimization.                            |
C+----------------------------------------------------------------------+
        Write(*,100)
        Write(*,110) IError
        Write(*,120) (DIRx(i),i=1,n)
        Write(*,130) DIRf
        Write(*,140) DIRmaxf
        Write(logfile,100) 
        Write(logfile,110) IError
        Write(logfile,120) (DIRx(i),i=1,n)
        Write(logfile,130) DIRf
        Write(logfile,140) DIRmaxf
C        diff=dtime(tarray)
C SP/2 Timing
        CALL CPU_TIME(T2)
        diff = T2-T1
        write(*,150) diff
        write(logfile,150) diff
C+----------------------------------------------------------------------+
C| Calculate the percent error.                                         |
C+----------------------------------------------------------------------+
        per2 = DIRf - fglobal
        if (fglobal .eq. 0.D0) then
           per2 = per2*100.D0
        else
           per2 = per2/ abs(fglobal)*100.D0
        end if
C+----------------------------------------------------------------------+
C| Save the results in the extra resultfile for use with Matlab etc.    |
C+----------------------------------------------------------------------+
        write(resultfile,200) problem, DIRmaxf, diff, per2, DIRalg
C+----------------------------------------------------------------------+
C| Close the logfile and rsultfile.                                     |
C+----------------------------------------------------------------------+
        close(logfile)
        close(resultfile)
      end if
C+----------------------------------------------------------------------+
C| Send signal to end program.                                          |
C+----------------------------------------------------------------------+
      CALL commexitIF(nprocs)

100   FORMAT('-------------- Final result ------------------')
110   FORMAT('DIRECT termination flag : ',I3)
120   FORMAT('DIRECT minimal point    : ',20f12.7)
130   FORMAT('DIRECT minimal value    : ',f12.7)
140   FORMAT('DIRECT number of f-eval : ',I5)
150   FORMAT('Time needed             : ',e10.4,' seconds.')
200   FORMAT(I3, ' ', I6, ' ', f12.4, ' ', e10.5, ' ',I6)
      end


C+----------------------------------------------------------------------+
C| Subroutine to read the values of the internal variables and the data |
C| for the interpolator.                                                |
C+----------------------------------------------------------------------+
      subroutine inputdata(n, u, l, logfile, DIReps, DIRmaxf,
     +       DIRmaxT, problem, Maxdim, DIRalg, fglobal, 
     + fglper, volper, sigmaper, 
     + iidata, iisize, ddata, idsize, cdata, icsize)
      implicit none
      Integer file, Maxdim, intproblem
      parameter(file = 31)
      Integer logfile,DOIFFCO      
C+----------------------------------------------------------------------+
C| DIRECT specific variables.                                           |
C+----------------------------------------------------------------------+
      Double Precision DIReps
      Integer DIRmaxf, DIRmaxT
      Double Precision fglobal, fglper, volper, sigmaper
      Integer DIRalg
C+----------------------------------------------------------------------+
C| Variables to pass user defined data to the function to be optimized. |
C+----------------------------------------------------------------------+
      INTEGER iisize, idsize, icsize
      INTEGER iidata(iisize)
      Double Precision ddata(idsize)
      Character*40 cdata(icsize)

C+----------------------------------------------------------------------+
C| General variables for the problem.                                   |
C+----------------------------------------------------------------------+
      integer n, i
      integer problem
      Double Precision l(maxdim), u(maxdim)
C+----------------------------------------------------------------------+
C| Variables to store the different file names.                         |
C+----------------------------------------------------------------------+
      character DIRectinit*20
      character problemdata*20

C+----------------------------------------------------------------------+
C| Read file names of the different files used in this run.             |
C+----------------------------------------------------------------------+
      open(unit = file, file='ini/main.ini')
      read(file, 154) DIRectinit
      read(file, 153) intproblem
      read(file, 153) DOIFFCO
      close(unit = file)
      problem = intproblem
C+----------------------------------------------------------------------+
C| Store the problem number in the last entry of iidata.                |
C+----------------------------------------------------------------------+
      iidata(iisize) = problem
      write(*,2000) 
      write(*,2010) DIRectinit
      if (logfile .gt. 0) then
        write(logfile,2000) 
        write(logfile,2010) DIRectinit
      end if
      open(unit = file, file='ini/problems.ini')
      read(file,153)
      do 40,i = 1,problem+1
         read(file, 154) problemdata
40    continue
      close(unit = file)
      write(*,2020) problemdata
      write(*,2030) problem
      if (logfile .gt. 0) then
        write(logfile,2020) problemdata
        write(logfile,2030) problem
      end if
C+----------------------------------------------------------------------+
C| Read DIRECT variables from DIRinit.ini                               |
C+----------------------------------------------------------------------+
      open(unit = file, file =  'ini/'//DIRectinit)
      read(file, 151) DIReps
      read(file, 150) DIRmaxf
      read(file, 150) DIRmaxT
      read(file, 150) DIRalg
C+----------------------------------------------------------------------+
C| Read in the percent error when DIRECT should stop. If the optimal    |
C| function value is not known (that is, when a real optimization is    |
C| done), set this value to 0 and fglobal to -1.D100. This ensures that |
C| the percentage condition cannot be satiesfied.                       |
C+----------------------------------------------------------------------+
      read(file, 151) fglper
C+----------------------------------------------------------------------+
C| Read in the percentage of the volume that the hyperrectangle which   |
C| assumes fmin at its center needs to have to stop. Set this value to  |
C| 0.D0 if you don't want to use this stopping criteria.                |
C+----------------------------------------------------------------------+
      read(file, 151) volper
C+----------------------------------------------------------------------+
C| Read in the bound on the measure that the hyperrectangle which       |
C| assumes fmin at its center needs to have to stop. Set this value to  |
C| 0.D0 if you don't want to use this stopping criteria.                |
C+----------------------------------------------------------------------+
      read(file, 151) sigmaper
      close(unit = file)
C+----------------------------------------------------------------------+
C| Read problem specifics from problem data file.                       |
C+----------------------------------------------------------------------+
      open(unit = file, file = 'problem/'//problemdata)
C+----------------------------------------------------------------------+
C| Read in the problem name. This name is used in the initial output    |
C| from DIRECT.                                                         |
C+----------------------------------------------------------------------+
      read(file,152) cdata(1)
      read(file, 150) n
C+----------------------------------------------------------------------+
C| Read in the (know) optimal function value. Note that this value is   |
C| generally not know, but for the test problems it is. If this value is|
C| unknown, set fglobal to -1.D100 and fglper (see above) to 0.         |
C+----------------------------------------------------------------------+
      read(file, 151) fglobal
      do 1000, i = 1,n
         read(file, 151) l(i)
1000  continue
      do 1005, i = 1,n
         read(file, 151) u(i)
1005  continue
      close(unit = file)
      
150   FORMAT(I10)
151   FORMAT(F20.10)
152   FORMAT(A40)
153   FORMAT(I20)
154   FORMAT(A20)

2000  FORMAT('Name of ini-directory    : ini/')
2010  FORMAT('Name of DIRect.ini file  : ',A20)
2020  FORMAT('Name of problemdata file : ',A20)
2030  FORMAT('Testproblem used         : ',I4)
      end 

C+----------------------------------------------------------------------+
C| Give out a header for the main program.                              |
C+----------------------------------------------------------------------+
      SUBROUTINE mainheader(logfile)
      IMPLICIT None
      Integer logfile

      write(*,100)
      write(*,110)
      write(*,120)
      write(*,130)
      write(*,140)
      write(*,150)
      write(*,160)
      write(*,170)
      write(*,180)
      write(*,190)
      write(*,200)
      if (logfile .gt. 0) then
        write(logfile,100)
        write(logfile,110)
        write(logfile,120)
        write(logfile,130)
        write(logfile,140)
        write(logfile,150)
        write(logfile,160)
        write(logfile,170)
        write(logfile,180)
        write(logfile,190)
        write(logfile,200)
      end if
100   FORMAT('+----------------------------------------+')
110   FORMAT('|       Example Program for DIRECT       |')
120   FORMAT('|  This program uses DIRECT to optimize  |')
130   FORMAT('|  testfunctions. Which testfunction is  |')
140   FORMAT('| optimized and what parameters are used |')
150   FORMAT('| is controlled by the files in ini/.    |')
160   FORMAT('|                                        |')
170   FORMAT('|     Owen Esslinger, Joerg Gablonsky,   |')
180   FORMAT('|             Alton Patrick              |')
190   FORMAT('|              04/15/2001                |')
200   FORMAT('+----------------------------------------+')
      end
