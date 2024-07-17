/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <mpi.h>
#include <malloc.h>
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printGhostMat2(const char mesg[], double *E, int m, int n);
void atomicPrintMat2(int myrank, int nprocs, int m, int n, double *E_prev);
double *alloc1D2(int m, int n);
void printArray(double* arr, int length);
void tileStats(double *E, int m, int n, int tileY, int tileX, double *_mx, double *sumSq);
extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 double finalMax, finalSumSq;
 int niter;
 int m = cb.m, n=cb.n;

 int nprocs = 1, myrank = 0;
 MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
 MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

 int y = cb.py, x = cb.px;
 int rows_per_proc = m / y;
 int cols_per_proc = n / x;
 int remainderY = m % y;
 int tileY = rows_per_proc + (myrank / x < remainderY ? 1 : 0); 
 int remainderX = n % x;
 int tileX = cols_per_proc + (myrank % x < remainderX ? 1 : 0);
 MPI_Datatype Y;
 MPI_Type_vector(tileY, 1, n + 2, MPI_DOUBLE, &Y);
 MPI_Type_commit(&Y);
 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
 for (niter = 0; niter < cb.niters; niter++){
    //   if  (cb.debug && (niter==0) && (myrank==0)){
	//   stats(E_prev,m,n,&mx,&sumSq);
    //       double l2norm = L2Norm(sumSq);
	//   repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	//   if (cb.plot_freq)
	//       plotter->updatePlot(E,  -1, m+1, n+1);
    //   }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
        MPI_Request send_reqs[4];
        MPI_Request recv_reqs[4];
        for(int i = 0; i < 4; i++){
            send_reqs[i] = MPI_REQUEST_NULL;
            recv_reqs[i] = MPI_REQUEST_NULL;
        }

        // 4 FOR LOOPS set up the padding needed for the boundary conditions
        int i,j;
        // Fills in the TOP Ghost Cells
        if (myrank < x){
            for (i = 1; i < tileX + 1; i++) {
                E_prev[i] = E_prev[i + (n+2)*2];
            }
            // printf("Process %d filled in the TOP Ghost Cells\n", myrank);
        } else {
            if (!cb.noComm){
                MPI_Isend(&E_prev[n+2+1], tileX, MPI_DOUBLE, myrank - x, 0, MPI_COMM_WORLD, &send_reqs[1]);
                // printf("Process %d sent to TOP Process %d \n", myrank, myrank - x);
                MPI_Irecv(&E_prev[1], tileX, MPI_DOUBLE, myrank - x, 0, MPI_COMM_WORLD, &recv_reqs[1]);
                // printf("Process %d received from TOP Process %d \n", myrank, myrank - x);
            }
        }

        if (myrank % x == (x - 1)) {
            //Fills in the RIGHT Ghost Cells
            for (i = (tileX+1)+(n+2); i <= (tileX+1)+(tileY)*(n+2); i+=(n+2)) {
                E_prev[i] = E_prev[i-2];
            }
        } else {
            if (!cb.noComm){
                MPI_Isend(&E_prev[tileX+(n+2)], 1, Y, myrank + 1, 0, MPI_COMM_WORLD, &send_reqs[2]);
                // printf("Process %d sent to RIGHT Process %d \n", myrank, myrank + 1);
                MPI_Irecv(&E_prev[tileX+(n+2)+1], 1, Y, myrank + 1, 0, MPI_COMM_WORLD, &recv_reqs[2]);
                // printf("Process %d received from RIGHT Process %d \n", myrank, myrank + 1);
            }
        }

        //Fills in the LEFT Ghost Cells
        if (myrank % x == 0) {
            for (i = n+2; i <= tileY*(n+2); i+=(n+2)) {
                E_prev[i] = E_prev[i+2];
            }	
        } else {
            if (!cb.noComm){
                MPI_Isend(&E_prev[n+2+1], 1, Y, myrank - 1, 0, MPI_COMM_WORLD, &send_reqs[3]);
                MPI_Irecv(&E_prev[n+2], 1, Y, myrank - 1, 0, MPI_COMM_WORLD, &recv_reqs[3]);
            }
        }
        // Fills in the BOTTOM Ghost Cells
        if (myrank >= nprocs - x && myrank < nprocs){
            for (i = (n+2)*(1+tileY)+1; i < (n+2)*(1+tileY)+1+tileX; i++) {
                E_prev[i] = E_prev[i - (n+2)*2];
            }
        } else {
            // printf("Process %d is sending to BOTTOM Process %d \n", myrank, myrank + x);
            // printArray(E_prev + (n+2)*tileY + 1, tileX);
            if (!cb.noComm){
                MPI_Isend(&E_prev[(n+2)*tileY+1], tileX, MPI_DOUBLE, myrank + x, 0, MPI_COMM_WORLD, &send_reqs[0]);
                // printf("Process %d sent to BOTTOM Process %d \n", myrank, myrank + x);
                MPI_Irecv(&E_prev[(n+2)*(tileY+1)+1], tileX, MPI_DOUBLE, myrank + x, 0, MPI_COMM_WORLD, &recv_reqs[0]);
                // printf("Process %d is receiving from BOTTOM Process %d \n", myrank, myrank + x);
            }
        }
        if (!cb.noComm){
            MPI_Waitall(4, send_reqs, MPI_STATUSES_IGNORE);
            MPI_Waitall(4, recv_reqs, MPI_STATUSES_IGNORE);
            // MPI_Barrier(MPI_COMM_WORLD);
            // printf("Process %d has finished filling in the Ghost Cells\n", myrank);
            // atomicPrintMat2(myrank, nprocs, m, n, E_prev);
        }

//////////////////////////////////////////////////////////////////////////////
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = innerBlockRowStartIndex + (tileY-1)*(n+2);
//  printf("Process %d innerBlockRowStartIndex: %d innerBlockRowEndIndex: %d\n", myrank, innerBlockRowStartIndex, innerBlockRowEndIndex);

#define FUSED 1

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        double* E_tmp = E + j;
	    double* E_prev_tmp = E_prev + j;
        double* R_tmp = R + j;
	for(i = 0; i < tileX; i++) {
	    double E_tmp_val = E_prev_tmp[i];
        double R_tmp_val = R_tmp[i];
        double E_prev_tmp_val = E_prev_tmp[i];

        E_tmp_val += alpha * (E_prev_tmp[i+1] + E_prev_tmp[i-1] - 4*E_prev_tmp_val + E_prev_tmp[i+(n+2)] + E_prev_tmp[i-(n+2)]);
        E_tmp_val += -dt * (kk * E_prev_tmp_val * (E_prev_tmp_val - a) * (E_prev_tmp_val - 1) + E_prev_tmp_val * R_tmp_val);
        R_tmp_val += dt * (epsilon + M1 * R_tmp_val / (E_prev_tmp_val + M2)) * (-R_tmp_val - kk * E_prev_tmp_val * (E_prev_tmp_val - b - 1));

        E_tmp[i] = E_tmp_val;
        R_tmp[i] = R_tmp_val;
        }
    }
#else

    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        double* E_tmp = E + j;
        double* E_prev_tmp = E_prev + j;
            for(i = 0; i < tileX; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        double* E_tmp = E + j;
        double* R_tmp = R + j;
	    double* E_prev_tmp = E_prev + j;
        for(i = 0; i < tileX; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////
    // MPI_Barrier(MPI_COMM_WORLD);
    // fprintf(stderr, "Processes finished iteration %d\n", niter);
    // atomicPrintMat2(myrank, nprocs, m, n, E_prev);
    // atomicPrintMat2(myrank, nprocs, m, n, R);
    // MPI_Barrier(MPI_COMM_WORLD);
   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        tileStats(E_prev, m, n, tileY,tileX, &mx, &sumSq);
        MPI_Reduce(&mx, &finalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sumSq, &finalSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myrank == 0) {
            Linf = finalMax;
            L2 = L2Norm(finalSumSq);
            repNorms(L2, Linf, dt, m, n, niter, cb.stats_freq);
        } 
     }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters
 if (!cb.noComm) {
    tileStats(E_prev, m, n, tileY, tileX, &mx, &sumSq);
    MPI_Reduce(&mx, &finalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sumSq, &finalSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myrank == 0) {
        Linf = finalMax;
        L2 = L2Norm(finalSumSq);
    } 
 } else {
    tileStats(E_prev, m, n, tileY, tileX, &mx, &sumSq);
    Linf = mx;
    L2 = L2Norm(sumSq);
 }

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}



void printMat2(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}

void atomicPrintMat2(int myrank, int nprocs, int m, int n, double *E_prev) {
    if (myrank == 0) {
        double *E_print;
        E_print = alloc1D2(m+2, n+2);
        printf("Process %d Printing:", 0);
        // printMat("", E_print, m, n);
        printGhostMat2("", E_prev, m, n);
    // Receive and print the matrices from the other processes
        for (int i = 1; i < nprocs; i++) {
            MPI_Recv(E_print, (m + 2) * (n + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d Printing:", i);
            // printMat("", E_print, m, n);
            printGhostMat2("", E_print, m, n);
        }
    } else {
    // Send the matrix to process 0
        MPI_Send(E_prev, (m + 2) * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

}



double *alloc1D2(int m, int n) {
    int nx = n, ny = m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * nx * ny));
    return(E);
}

void printGhostMat2(const char mesg[], double *E, int m, int n) {
    int i;
    printf("%s\n", mesg);
    for (i = 0; i < (m + 2) * (n + 2); i++) {
        int rowIndex = i / (n + 2);
        int colIndex = i % (n + 2);
        if ((colIndex >= 0) && (colIndex <= n + 1))
            if ((rowIndex >= 0) && (rowIndex <= m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}

void printArray(double* arr, int length) {
    for (int i = 0; i < length; i++) {
        printf("%6.3f ", arr[i]);
    }
    printf("\n");
}

void tileStats(double *E, int m, int n, int tileY, int tileX, double *_mx, double *sumSq){
     double mx = -1;
     double _sumSq = 0;
     int i, j;

     for (i=0; i<(tileY+2)*(n+2); i++) {
        int rowIndex = i / (n+2);			// gives the current row number in 2D array representation
        int colIndex = i % (n+2);		// gives the base index (first row's) of the current index		

        if(colIndex == 0 || colIndex > tileX || rowIndex == 0 || rowIndex > tileY)
            continue;

        _sumSq += E[i]*E[i];
        double fe = fabs(E[i]);
        if (fe > mx)
            mx = fe;
    }
    *_mx = mx;
    *sumSq = _sumSq;
}
