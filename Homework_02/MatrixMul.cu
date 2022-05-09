#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#define rowx 1024
#define colx 256
#define rowy 256
#define coly 512
#define blocks 32

__global__ void matrix_multi_gpu(int *M, int *N, int *P) 
{
	__shared__ int shareA[blocks][blocks];
	__shared__ int shareB[blocks][blocks];
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int maxx = ceil((double)colx / blocks);
	int sum = 0;
	for (int step = 0; step < maxx; step++) {
		if (i < rowx && (threadIdx.y + blockDim.y * step) < colx)
			shareA[threadIdx.x][threadIdx.y] = M[i * colx + (threadIdx.y + blockDim.y * step)];
		else
			shareA[threadIdx.x][threadIdx.y] = 0;
		if ((threadIdx.x + blockDim.y * step) < rowy && j < coly)
			shareB[threadIdx.y][threadIdx.x] = N[(threadIdx.x + blockDim.y * step) * coly + j];
		else
			shareB[threadIdx.y][threadIdx.x] = 0;
		__syncthreads();
		for (int k = 0; k < blocks; k++)
			sum += shareA[threadIdx.x][k] * shareB[threadIdx.y][k];
		__syncthreads();
	}
	if (i < rowx && j < coly)
		P[i * coly + j] = sum;
}

void matrix_multi_cpu(int *a, int *b, int *c) 
{
	for (int x = 0; x < rowx; x++) 
	{
		for (int y = 0; y < coly; y++) 
		{
			int s = 0;
			for (int i = 0; i < colx; i++) 
			{
				s += a[x * colx + i] * b[i * coly + y];
			}
			c[x * coly + y] = s;
		}
	}
}

void print(int *A, int *B, int *C) 
{
	printf("matrixA = \n");
	for (int i = 0; i < rowx; i++) 
	{
		for (int j = 0; j < colx; j++)
			printf("%d ", A[i * colx + j]);
		printf("\n");
	}
	printf("matrixB = \n");
	for (int i = 0; i < rowy; i++) 
	{
		for (int j = 0; j < coly; j++)
			printf("%d ", B[i * coly + j]);
		printf("\n");
	}
	printf("matrixC = matrixA * matrixB = \n");
	for (int i = 0; i < rowx; i++) 
	{
		for (int j = 0; j < coly; j++)
			printf("%d ", C[i * coly + j]);
		printf("\n");
	}
}

int CheckAnswer(int* _C, int* _D, int size)
{
    int result = 1;
	for (int i = 0; i < size && result == 1; ++i)
    {
        if (_C[i] != _D[i])
            result = 0;
    }
    return result;
}

int main() 
{
	int *A = (int *)malloc(sizeof(int) * rowx * colx);
	int *B = (int *)malloc(sizeof(int) * rowy * coly);
	int *C_gpu = (int *)malloc(sizeof(int) * rowx * coly);
	int *C_cpu = (int *)malloc(sizeof(int) * rowx * coly);
	int *d_dataA, *d_dataB, *d_dataC;
	
	cudaMalloc((void **)&d_dataA, sizeof(int) * rowx * colx);
	cudaMalloc((void **)&d_dataB, sizeof(int) * rowy * coly);
	cudaMalloc((void **)&d_dataC, sizeof(int) * rowx * coly);
	
	srand((unsigned)time(NULL));
	for (int i = 0; i < rowx * colx; i++)
		A[i] = rand() % 100 + 1;
	for (int i = 0; i < rowy * coly; i++)
		B[i] = rand() % 100 + 1;
	
	struct timeval start_gpu, end_gpu;
	gettimeofday(&start_gpu, NULL);
	cudaMemcpy(d_dataA, A, sizeof(int) * rowx * colx, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataB, B, sizeof(int) * rowy * coly, cudaMemcpyHostToDevice);
	dim3 block_size(blocks, blocks);
	int maxr = rowx > rowy ? rowx : rowy, maxc = colx > coly ? colx : coly;
	int gridx = ceil((double)maxr / block_size.x), gridy = ceil((double)maxc / block_size.y);
	dim3 grid_size(gridx, gridy);
	matrix_multi_gpu <<< grid_size, block_size >>> (d_dataA, d_dataB, d_dataC);
	cudaMemcpy(C_gpu, d_dataC, sizeof(int) * rowx * coly, cudaMemcpyDeviceToHost);
	gettimeofday(&end_gpu, NULL);
	
	struct timeval start_cpu, end_cpu;
	gettimeofday(&start_cpu, NULL);
	matrix_multi_cpu(A, B, C_cpu);
	gettimeofday(&end_cpu, NULL);

    if (CheckAnswer(C_gpu, C_cpu, rowx * coly))
	        printf("The answer is right!\n");
	else
	        printf("The answer is wrong!\n");
	
	free(A);
	free(B);
	free(C_gpu);
	free(C_cpu);
	cudaFree(d_dataA);
	cudaFree(d_dataB);
	cudaFree(d_dataC);
	
	int timeuse_gpu = 1000000 * (end_gpu.tv_sec - start_gpu.tv_sec) + end_gpu.tv_usec - start_gpu.tv_usec;
	int timeuse_cpu = 1000000 * (end_cpu.tv_sec - start_cpu.tv_sec) + end_cpu.tv_usec - start_cpu.tv_usec;
	printf("GPU运行时间为%lfs\n", (double)timeuse_gpu / (double)1000000);
	printf("CPU运行时间为%lfs\n", (double)timeuse_cpu / (double)1000000);
	
	return 0;
}
