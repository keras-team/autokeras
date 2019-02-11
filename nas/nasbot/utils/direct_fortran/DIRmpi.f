C+-----------------------------------------------------------------------+
C| Program       : DIRect.f (subroutine DIRmpi.f)                        |
C| Last modified : 04-13-2001                                            |
C| Written by    : Alton Patrick, Joerg Gablonsky                        |
C| Routines to use the MPI interface for parallel programming. These     |
C| interface routines were developed by Alton Patrick for IFFCO.         |
C+-----------------------------------------------------------------------+


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

      include 'mpif.h'


C Arguments
      integer tid, tag, n, flag, k
      double precision fscale, u(n), l(n), x(n)

C Local variables
      integer mx, i, istat
      parameter(mx=24)
      double precision message(3+3*mx)                


C     Put all data into an array of doubles for easy passing
      message(1) = dble(flag)
      message(2) = dble(k)
      message(3) = fscale

      do 10 i = 1,n
        message(3+i) = u(i)
        message(3+n+i) = l(i)
        message(3+n+n+i) = x(i)
 10   continue

      call mpi_send(message, 3+3*n, mpi_double_precision, tid, tag, 
     *  mpi_comm_world, istat)


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

      include 'mpif.h'


C Arguments
      integer tid, tag, n, flag, k
      double precision fscale, u(n), l(n), x(n)

C Local Variables
      integer mx, i, localtid, localtag
      parameter(mx=24)
      double precision message(3+3*mx)
      integer istat

      integer status(mpi_status_size)


      localtid = tid
      localtag = tag


      if(tid.eq.-1) then
        localtid = mpi_any_source
      end if
      if(tag.eq.-1) then
        localtag = mpi_any_tag
      end if

      call mpi_recv(message, 3+3*n, mpi_double_precision, localtid, 
     *  localtag, mpi_comm_world, status, istat)

C     Get the components of the message out of the vector of doubles
      flag = int(message(1))
      k = int(message(2))
      fscale = message(3)
      
      do 10 i = 1,n
        u(i) = message(3+i)
        l(i) = message(3+n+i)
        x(i) = message(3+n+n+i)
 10   continue


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

      include 'mpif.h'


C Arguments
      integer tid, tag, k, mytid, fflag
      double precision f

C Local variables
      integer i, istat
      double precision message(4)                


C     Put all data into an array of doubles for easy passing
      message(1) = dble(k)
      message(2) = dble(mytid)
      message(3) = f
      message(4) = dble(fflag)

      call mpi_send(message, 4, mpi_double_precision, tid, tag, 
     *  mpi_comm_world, istat)


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

      include 'mpif.h'


C Arguments
      integer tid, tag, k, slavetid, fflag
      double precision f

C     Local Variables
      integer i, localtid, localtag
      double precision message(4)
      integer istat

      integer status(mpi_status_size)


      localtid = tid
      localtag = tag


      if(tid.eq.-1) then
        localtid = mpi_any_source
      end if
      if(tag.eq.-1) then
        localtag = mpi_any_tag
      end if
      call mpi_recv(message, 4, mpi_double_precision, localtid, 
     *  localtag, mpi_comm_world, status, istat)

C     Get the components of the message out of the vector of doubles
      k = int(message(1))
      slavetid = int(message(2))
      f = message(3)
      fflag = int(message(4))

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

      include 'mpif.h'


C Arguments
      integer mytid

C Local Variables
      integer istat


      call mpi_comm_rank(mpi_comm_world, mytid, istat)


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

      include 'mpif.h'


C Arguments
      integer nprocs

C Local Variables
      integer istat


      call mpi_comm_size(mpi_comm_world, nprocs, istat) 


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

      include 'mpif.h'


C Arguments
      integer i, tid

C Local Variables
      integer temp


      tid = i


      end

C-----------------------------------------------------------------
C c8) comminitIF
C
C Initializes communication.  
C-----------------------------------------------------------------
      subroutine comminitIF()
      implicit none

C     PVM or MPI header file.  Path may be different on other systems.

      include 'mpif.h'


C Local Variables
      integer inum, istat


      call mpi_init(istat)


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

      include 'mpif.h'


C Arguments
      integer nprocs

C Local Variables
      integer istat

C     Do not remove this output statement; the delay is necessary on
C     some platforms to prevent one or more processes from exiting without
C     finalizing.
      write(*,*) ' '


      call mpi_barrier(mpi_comm_world)
      call mpi_finalize(istat)


      end
