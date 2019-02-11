C+-----------------------------------------------------------------------+
C| Program       : DIRect.f (subroutine DIRpvm.f)                        |
C| Last modified : 04-13-2001                                            |
C| Written by    : Alton Patrick, Joerg Gablonsky                        |
C| Routines to use the PVM interface for parallel programming. These     |
C| interface routines were developed by Alton Patrick for IFFCO.         |
C+-----------------------------------------------------------------------+


C The following parallel communication routines are only used in the
C parallel versions.

C-----------------------------------------------------------------
C c1) mastersendIF
C
C This routine sends the standard message from the master processor
C to a slave, using either PVM or MPI calls.
C
C Input variables:
C   tid -- processor to send to
C   tag -- message tag
C   n -- dimension of problem
C   flag -- action flag for slave
C   k -- extra info for slave
C   fscale -- function scaling
C   u -- upper bounds
C   l -- lower bounds
C   x -- point to evaluate objective func. at
C-----------------------------------------------------------------
      subroutine mastersendIF(tid, tag, n, flag, k, fscale, u, l, x)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer tid, tag, n, flag, k
      double precision fscale, u(n), l(n), x(n)

C Local variables
      integer mx, i, istat
      parameter(mx=24)
      double precision message(3+3*mx)                


      call pvmfinitsend(PvmDataRaw,istat)
      call pvmfpack(integer4,flag,1,1,istat)
      call pvmfpack(integer4,k,1,1,istat)
      call pvmfpack(real8,fscale,1,1,istat)
      call pvmfpack(real8,u,n,1,istat)
      call pvmfpack(real8,l,n,1,istat)
      call pvmfpack(real8,x,n,1,istat)
      call pvmfsend(tid,tag,istat)



      end


C-----------------------------------------------------------------
C c2) slaverecvIF
C
C This routine receives the standard message sent from the master
C to a slave, using either PVM or MPI calls.
C Input variable:
C   tid -- processor to receive from
C   tag -- message tag 
C   n -- problem dimension
C Output variables:
C   flag -- slave action flag
C   k -- extra data for slave
C   fscale -- function scaling
C   u -- upper bounds
C   l -- lower bounds
C   x -- point to evaluate objective func. at 
C-----------------------------------------------------------------
      subroutine slaverecvIF(tid, tag, n, flag, k, fscale, u, l, x)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer tid, tag, n, flag, k
      double precision fscale, u(n), l(n), x(n)

C Local Variables
      integer mx, i, localtid, localtag
      parameter(mx=24)
      double precision message(3+3*mx)
      integer istat


      localtid = tid
      localtag = tag


      call pvmfrecv(localtid,localtag,istat)
      call pvmfunpack(integer4,flag,1,1,istat)
      call pvmfunpack(integer4,k,1,1,istat)
      call pvmfunpack(real8,fscale,1,1,istat)
      call pvmfunpack(real8,u,n,1,istat)
      call pvmfunpack(real8,l,n,1,istat)
      call pvmfunpack(real8,x,n,1,istat)



      end


C-----------------------------------------------------------------
C c3) slavesendIF
C
C This routine sends the standard message from a slave processor
C to the master, using either PVM or MPI calls.
C
C Input variables:
C   tid -- processor to send to
C   tag -- message tag
C   k -- extra info from master
C   mytid -- tid of slave sending message
C   f -- function value
C-----------------------------------------------------------------
      subroutine slavesendIF(tid, tag, k, mytid, f, fflag)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer tid, tag, k, mytid, fflag
      double precision f

C Local variables
      integer i, istat
      double precision message(4)                


      call pvmfinitsend(PvmDataRaw,istat)
      call pvmfpack(integer4,k,1,1,istat)
      call pvmfpack(integer4,mytid,1,1,istat)
      call pvmfpack(real8,f,1,1,istat)
      call pvmfpack(integer4,fflag,1,1,istat)
      call pvmfsend(tid,tag,istat)



      end


C-----------------------------------------------------------------
C c4) masterrecvIF
C
C This routine receives the standard message sent from a slave
C to the master, using either PVM or MPI calls.
C Input variable:
C   tid -- processor to receive from
C   tag -- message tag 
C Output variables:
C   k -- extra data 
C   slavetid -- tid of slave message is from
C   f -- function value
C-----------------------------------------------------------------
      subroutine masterrecvIF(tid, tag, k, slavetid, f, fflag)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer tid, tag, k, slavetid, fflag
      double precision f

C     Local Variables
      integer i, localtid, localtag
      double precision message(4)
      integer istat


      localtid = tid
      localtag = tag


      call pvmfrecv(localtid, localtag, istat)
      call pvmfunpack(integer4, k, 1, 1, istat)
      call pvmfunpack(integer4, slavetid, 1, 1, istat)
      call pvmfunpack(real8, f, 1, 1, istat)
      call pvmfunpack(integer4, fflag, 1, 1, istat)


      end

C-----------------------------------------------------------------
C c5) getmytidIF
C
C Gets the task id or rank of this process.
C Output variable:
C   mytid -- task id or rank of this process.
C-----------------------------------------------------------------
      subroutine getmytidIF(mytid)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer mytid

C Local Variables
      integer istat


      call pvmfmytid(mytid)


      end


C-----------------------------------------------------------------
C c6) getnprocsIF
C
C Returns the number of processes available.
C Output variable:
C   nprocs -- number of processes available.
C-----------------------------------------------------------------
      subroutine getnprocsIF(nprocs)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer nprocs

C Local Variables
      integer istat


      call pvmfgsize('iffcogroup', nprocs)


      end

C-----------------------------------------------------------------
C c7) gettidIF
C
C Returns the task id or rank of a process given that process'
C "logical" designation.  I.e. if i = 0, returns the tid (PVM) 
C or the rank (MPI) of the first process.  Note that in the MPI
C case, this subroutine simply returns i.
C Input variable:
C   i -- "logical" designation of a process.
C Output variable:
C   tid -- task id or rank of the specified process.
C-----------------------------------------------------------------
      subroutine gettidIF(i, tid)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer i, tid

C Local Variables
      integer temp


      call pvmfgettid('iffcogroup', i, tid)


      end

C-----------------------------------------------------------------
C c8) comminitIF
C
C Initializes communication.  
C-----------------------------------------------------------------
      subroutine comminitIF()
      implicit none

C     PVM or MPI header file.  Path may be different on other systems.

      include 'fpvm3.h'


C Local Variables
      integer inum, istat


      call pvmfjoingroup('iffcogroup', inum)


      end

C-----------------------------------------------------------------
C c9) commexitIF
C
C Shuts down communication.
C Input variable:
C   n -- number of processes
C-----------------------------------------------------------------
      subroutine commexitIF(nprocs)
      implicit none

C     PVM or MPI header file.

      include 'fpvm3.h'


C Arguments
      integer nprocs

C Local Variables
      integer istat

C     Do not remove this output statement; the delay is necessary on
C     some platforms to prevent one or more processes from exiting without
C     finalizing.
      write(*,*) ' '


      call pvmfbarrier('iffcogroup', nprocs, istat)
      call pvmfexit(istat)


      end
