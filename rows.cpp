#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int ProcNum = 0;
int ProcRank = 0;

void RandomDataInitialization(double *pAMatrix, double *pBMatrix, int Size) {
    int i, j;
    srand(unsigned(clock()));
    for (i = 0; i < Size; i++)
        for (j = 0; j < Size; j++) {
            pAMatrix[i * Size + j] = rand() / double(1000);
            pBMatrix[i * Size + j] = rand() / double(1000);
        }
}

void RowsMultiplication(double *pArows, double *pBrows, double *pCrows, int Size, int RowNum, int index) {
    int i, j, k;
    for (i = 0; i < RowNum; i++) {
        for (j = 0; j < Size; j++)
            for (k = 0; k < RowNum; k++)
                pCrows[i * Size + j] += pArows[i * Size + index * RowNum + k] * pBrows[k * Size + j];
    }
}

void ProcessInitialization(double *&pAMatrix, double *&pBMatrix, double *&pCMatrix, double *&pArows, double *&pBrows,
                           double *&pCrows, int &Size, int &RowNum) {
    if (ProcRank == 0) {
        do {
            printf("\nEnter size of the initial objects: ");
            scanf("%d", &Size);
        } while (Size < 0);
    }
    int RestRows;
    int i;

    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    RestRows = Size;
    for (i = 0; i < ProcRank; i++)
        RestRows = RestRows - RestRows / (ProcNum - i);
    RowNum = RestRows / (ProcNum - ProcRank);

    pArows = new double[RowNum * Size];
    pBrows = new double[RowNum * Size];
    pCrows = new double[RowNum * Size];

    for (int i = 0; i < RowNum * Size; i++) {
        pCrows[i] = 0;
    }
    pCMatrix = new double[Size * Size];

    if (ProcRank == 0) {
        pAMatrix = new double[Size * Size];
        pBMatrix = new double[Size * Size];

        RandomDataInitialization(pAMatrix, pBMatrix, Size);
    }
}

void DataDistribution(double *pAMatrix, double *pBMatrix, double *pArows, double *pBrows, int Size, int RowNum) {

    int *pSendNum;
    int *pSendInd;
    int RestRows = Size;

    pSendInd = new int[ProcNum];
    pSendNum = new int[ProcNum];

    RowNum = (Size / ProcNum);
    pSendNum[0] = RowNum * Size;
    pSendInd[0] = 0;
    for (int i = 1; i < ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows / (ProcNum - i);
        pSendNum[i] = RowNum * Size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }

    MPI_Scatterv(pAMatrix, pSendNum, pSendInd, MPI_DOUBLE, pArows, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(pBMatrix, pSendNum, pSendInd, MPI_DOUBLE, pBrows, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pSendNum;
    delete[] pSendInd;

}

void ResultCollection(double *pCMatrix, double *pCrows, int Size, int RowNum) {

    int i;
    int *pReceiveNum;
    int *pReceiveInd;  /* Index of the first element from current process
						  in result vector */
    int RestRows = Size * Size;

    pReceiveNum = new int[ProcNum];
    pReceiveInd = new int[ProcNum];

    pReceiveInd[0] = 0;
    pReceiveNum[0] = Size * RowNum;
    for (i = 1; i < ProcNum; i++) {
        RestRows -= pReceiveNum[i - 1];
        pReceiveNum[i] = RestRows / (ProcNum - i);
        pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
    }

    MPI_Allgatherv(pCrows, pReceiveNum[ProcRank], MPI_DOUBLE, pCMatrix,
                   pReceiveNum, pReceiveInd, MPI_DOUBLE, MPI_COMM_WORLD);

    delete[] pReceiveNum;
    delete[] pReceiveInd;
}


void BRowsCommunication(double *pBrows, int Size, int RowNum) {
    MPI_Status Status;
    int NextProc = ProcRank + 1;
    if (ProcRank == ProcNum - 1) NextProc = 0;
    int PrevProc = ProcRank - 1;
    if (ProcRank == 0) PrevProc = ProcNum - 1;

    MPI_Sendrecv_replace(pBrows, Size * RowNum, MPI_DOUBLE,
                         NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
}

void ParallelResultCalculation(double *pArows, double *pBrows, double *pCrows, int Size, int RowNum) {

    int index = 0;
    for (int iter = 0; iter < ProcNum; iter++) {
        if (iter == 0) index = ProcRank;
        else {
            index--;
            if (index == -1)
                index = ProcNum - 1;
        }

        RowsMultiplication(pArows, pBrows, pCrows, Size, RowNum, index);

        BRowsCommunication(pBrows, Size, RowNum);
    }
}

void ProcessTermination(double *pAMatrix, double *pBMatrix, double *pCMatrix, double *pAblock, double *pBblock,
                        double *pCblock) {
    if (ProcRank == 0) {
        delete[] pAMatrix;
        delete[] pBMatrix;
        delete[] pCMatrix;
    }
    delete[] pAblock;
    delete[] pBblock;
    delete[] pCblock;
}

int main(int argc, char *argv[]) {
    double *pAMatrix;
    double *pBMatrix;
    double *pCMatrix;
    int Size;
    double *pArows;
    double *pBrows;
    double *pCrows;
    int RowNum;
    double Start, Finish, Duration;

    setvbuf(stdout, 0, _IONBF, 0);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) printf("Parallel matrix multiplication program\n");

    ProcessInitialization(pAMatrix, pBMatrix, pCMatrix, pArows, pBrows, pCrows, Size, RowNum);

    Start = MPI_Wtime();
    DataDistribution(pAMatrix, pBMatrix, pArows, pBrows, Size, RowNum);
    ParallelResultCalculation(pArows, pBrows, pCrows, Size, RowNum);
    ResultCollection(pCMatrix, pCrows, Size, RowNum);
    Finish = MPI_Wtime();

    Duration = Finish - Start;

    if (ProcRank == 0) printf("Time of execution = %f\n", Duration);

    ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pArows, pBrows, pCrows);

    MPI_Finalize();

}