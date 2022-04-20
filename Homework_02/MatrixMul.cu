#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include <math.h>

const int Row = 1280;
const int Col = 1280;
 
__global__ void matrix_mul_gpu(float *M, float* N, float* P, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
                
    float sum = 0.0;
    for(int k=0;k<width;k++)
    {
        float a = M[j*width+k];
        float b = N[k*width+i];
        sum += a*b;
    }
    P[j*width+i] = sum;
}

void matrix_mul_cpu(float* M, float* N, float* P, int width)
{
    for(int i=0;i<width;i++)
        for(int j=0;j<width;j++)
        {
            float sum = 0.0;
            for(int k=0;k<width;k++)
            {
                float a = M[i*width+k];
                float b = N[k*width+j];
                sum += a*b;
            }
            P[i*width+j] = sum;
        }
}
 
int main()
{
    struct timeval start, end;

    float *A = (float *)malloc(sizeof(float) * Row * Col);
    float *B = (float *)malloc(sizeof(float) * Row * Col);
    float *C = (float *)malloc(sizeof(float) * Row * Col);
    //malloc device memory
    float *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(float) *Row*Col);
    cudaMalloc((void**)&d_dataB, sizeof(float) *Row*Col);
    cudaMalloc((void**)&d_dataC, sizeof(float) *Row*Col);
    //set value
    for (int i = 0; i < Row*Col; i++) {
        A[i] = 90;
        B[i] = 10;
    }
                                                                
    cudaMemcpy(d_dataA, A, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, B, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);
    dim3 threadPerBlock(16, 16);
    dim3 blockNumber((Col+threadPerBlock.x-1)/ threadPerBlock.x, (Row+threadPerBlock.y-1)/ threadPerBlock.y );
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);
	
	gettimeofday( &start, NULL );
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);

    //拷贝计算数据-一级数据指针
    cudaMemcpy(C, d_dataC, sizeof(float) * Row * Col, cudaMemcpyDeviceToHost);
	gettimeofday( &end, NULL );
    int GPUtimeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("GPU total time is %d ms\n", GPUtimeuse/1000);
	
	gettimeofday( &start, NULL );
	matrix_mul_cpu(A, B, C, Col);
	gettimeofday( &end, NULL );
	int CPUtimeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	printf("CPU total time is %d ms\n", CPUtimeuse/1000);
                                                                                             
    //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);
    return 0;
}
