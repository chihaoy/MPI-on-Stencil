/*
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
 // Needed for memalign
#include <malloc.h>
#include <mpi.h>
#include "cblock.h"
using namespace std;
extern control_block cb;

void printMat(const char mesg[], double *E, int m, int n);
void printGhostMat(const char mesg[], double *E, int m, int n);
void atomicPrintMat(int myrank, int nprocs, int m, int n, double *E_prev);
double *alloc1D(int m, int n);

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init(double *E, double *E_prev, double *R, int m, int n) {
    int nprocs = 1, myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

// blocking factor
    int y = cb.py, x = cb.px;
    int rows_per_proc = m / y;
    int cols_per_proc = n / x;
    int remainderY = m % y;
    int tileY = rows_per_proc + (myrank / x < remainderY ? 1 : 0); 
    int remainderX = n % x;
    int tileX = cols_per_proc + (myrank % x < remainderX ? 1 : 0);
    int tileYtoSend = rows_per_proc + (m % y  > 0? 1 : 0);
    int tileXtoSend = cols_per_proc + (n % x > 0? 1 : 0);

// synchronization
    MPI_Request send_reqs[2 * (nprocs - 1)];
    MPI_Request recv_reqs[2];

// Initialize the MPI data type for sending the initial conditions
    MPI_Datatype tile;
    MPI_Type_vector(tileYtoSend, tileXtoSend, n + 2, MPI_DOUBLE, &tile);
    MPI_Type_commit(&tile);

    if (myrank == 0) {
        // printf("Process 0 init: tileYtoSend = %d, tileXtoSend = %d\n", tileYtoSend, tileXtoSend);
        // Initialize E_prev
        int i;

        for (i = 0; i < (m + 2) * (n + 2); i++)
            E_prev[i] = R[i] = 0;

        for (i = (n + 2); i < (m + 1) * (n + 2); i++) {
            int colIndex = i % (n + 2);		// gives the base index (first row's) of the current index

            // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || colIndex < ((n + 1) / 2 + 1))
                continue;

            E_prev[i] = 1.0;
        }
        // Initialize R
        for (int i = 0; i < (m + 2) * (n + 2); i++) {
            int rowIndex = i / (n + 2);		// gives the current row number in 2D array representation
            int colIndex = i % (n + 2);		// gives the base index (first row's) of the current index

            // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
            if (colIndex == 0 || colIndex == (n + 1) || rowIndex < ((m + 1) / 2 + 1))
                continue;

            R[i] = 1.0;
        }

        // Send the initial conditions to the worker processes
        for (int i = 1; i < nprocs; i++) {
            // MPI_Datatype tile;
            // int tileYtoSend = rows_per_proc + (i / x < remainderY ? 1 : 0);
            // int tileXtoSend = cols_per_proc + (i % x < remainderX ? 1 : 0);
            // MPI_Type_vector(tileYtoSend, tileXtoSend, n + 2, MPI_DOUBLE, &tile);
            // MPI_Type_commit(&tile);
            int tileY = rows_per_proc + (i / x < remainderY ? 1 : 0);
            int tileX = cols_per_proc + (i % x < remainderX ? 1 : 0);
            int start_row, start_col;
            if (tileX < tileXtoSend) {
                start_col = i % x * cols_per_proc + ((i % x < remainderX) ? i % x : remainderX);
            } else {
                start_col = i % x * cols_per_proc + ((i % x < remainderX) ? i % x : remainderX) + 1;
            }

            if (tileY < tileYtoSend) {
                start_row = i / x * rows_per_proc + ((i / x < remainderY) ? i / x : remainderY);
            } else {
                start_row = i / x * rows_per_proc + ((i / x < remainderY) ? i / x : remainderY) + 1;
            }
            // int start_row = 1 + i / x * rows_per_proc + ((i / x < remainderY) ? i / x : remainderY);
            // int start_col = 1 + i % x * cols_per_proc + ((i % x < remainderX) ? i % x : remainderX);
            // printf("Process 0 init sending to process %d: start_row = %d, start_col = %d\n", i, start_row, start_col);
            // printf("Process 0 init sending to process %d: start_row = %d, end_row = %d\n", i, start_row, end_row);
            // print send portion
            MPI_Isend(&E_prev[start_row * (n + 2) + start_col], 1, tile, i, 0, MPI_COMM_WORLD, &send_reqs[2 * (i - 1)]);
            MPI_Isend(&R[start_row * (n + 2) + start_col], 1, tile, i, 0, MPI_COMM_WORLD, &send_reqs[2 * (i - 1) + 1]);
            // MPI_Isend(&E_prev[start_row * (n + 2) + start_col], tileXtoSend * tileYtoSend, tile, i, 0, MPI_COMM_WORLD, &send_reqs[2 * (i - 1)]);
            // MPI_Isend(&R[start_row * (n + 2 + start_col], tileXtoSend * tileYtoSend, tile, i, 0, MPI_COMM_WORLD, &send_reqs[2 * (i - 1) + 1]);
            // MPI_Type_free(&tile);
        }
        MPI_Waitall((nprocs - 1) * 2, send_reqs, MPI_STATUSES_IGNORE);
        // printf("Process 0 FINISH sending initial conditions to all worker processes\n");
    }
    else {
        // Receive the initial conditions from process 0
        // printf("Process %d expecting tileY: %d tileX: %d\n", myrank, tileY, tileX);
        // MPI_Datatype tile;
        // MPI_Type_vector(tileY, tileX, n + 2, MPI_DOUBLE, &tile);
        // MPI_Type_commit(&tile);
        int start_col, start_row;
        if (tileX < tileXtoSend) {
            start_col = 0;
        } else {
            start_col = 1;
        }

        if (tileY < tileYtoSend) {
            start_row = 0;
        } else {
            start_row = 1;
        }

        MPI_Recv(E_prev + (n + 2) * start_row + start_col, 1, tile, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(R + (n + 2) * start_row + start_col, 1, tile, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // MPI_Type_free(&tile);
        // printf("Process %d FINISH receiving initial conditions from process 0\n", myrank);
    }
    MPI_Type_free(&tile);

    // We only print the meshes if they are small enough
#if 1
    // printMat("Process 0 Priting: Initial E_prev", E_prev, m, n);
    // printMat("Process 0 Priting: Initial R", R, m, n);
#endif
    // atomicPrintMat(myrank, nprocs, m, n, E_prev);
    // atomicPrintMat(myrank, nprocs, m, n, R);
}

void atomicPrintMat(int myrank, int nprocs, int m, int n, double *E_prev) {
    if (myrank == 0) {
        double *E_print;
        E_print = alloc1D(m+2, n+2);
        printf("Process %d Printing:", 0);
        // printMat("", E_print, m, n);
        printGhostMat("", E_prev, m, n);
    // Receive and print the matrices from the other processes
        for (int i = 1; i < nprocs; i++) {
            MPI_Recv(E_print, (m + 2) * (n + 2), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d Printing:", i);
            // printMat("", E_print, m, n);
            printGhostMat("", E_print, m, n);
        }
    } else {
    // Send the matrix to process 0
        MPI_Send(E_prev, (m + 2) * (n + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

}

double *alloc1D(int m, int n) {
    int nx = n, ny = m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * nx * ny));
    return(E);
}

void printGhostMat(const char mesg[], double *E, int m, int n) {
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

void printMat(const char mesg[], double *E, int m, int n) {
    int i;
#if 0
    if (m > 8)
        return;
#else
    if (m > 34)
        return;
#endif
    printf("%s\n", mesg);
    for (i = 0; i < (m + 2) * (n + 2); i++) {
        int rowIndex = i / (n + 2);
        int colIndex = i % (n + 2);
        if ((colIndex > 0) && (colIndex < n + 1))
            if ((rowIndex > 0) && (rowIndex < m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}
