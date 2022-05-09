#include<stdio.h>
#include<cuda_runtime.h>
#define BLOCKNUM 16384
#define THREADNUM 512

__global__ void Ginit(int *a,int *b)
{
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    a[tid]=blockIdx.x;
    b[tid]=blockIdx.x+1;   
}

__global__ void Gmultiply(int* a,int *b)
{
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    a[tid]=a[tid]*b[tid];
} 

void CpuMul(int*a,int *b)
{
    for(int i=0;i<BLOCKNUM;i++)
	{
        for(int j=0;j<THREADNUM;j++)
		{
            int tid=i*THREADNUM+j;
            a[tid]=j;
            b[tid]=j+1;
            a[tid]=a[tid]*b[tid];
        }
    }
	printf("CPU Multiply Finished.\n");
}

int main()
{
    int* a;
    int* b;
    cudaSetDevice(1);
    cudaMallocManaged(&a,BLOCKNUM*THREADNUM*sizeof(int));
    cudaMallocManaged(&b,BLOCKNUM*THREADNUM*sizeof(int));
    Ginit<<<BLOCKNUM,THREADNUM>>>(a,b);
	printf("GPU Initialization Finished.\n");
    cudaDeviceSynchronize();
    Gmultiply<<<BLOCKNUM,THREADNUM>>>(a,b);
	printf("GPU Multiply Finished.\n");
    cudaDeviceSynchronize();
    CpuMul(a,b);
    cudaFree(a);
    cudaFree(b);
    return 0;
} 
