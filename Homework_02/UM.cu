#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void InitData(float* data, size_t nElem)
{
	for (size_t i = 0; i < nElem; i++)
	{
		data[i] = i % 255;
	}
}

void SumArraysOnHost(float* h_A, float* h_B, float* hostRef, size_t nElem)
{
	for (size_t i = 0; i < nElem; i++)
	{
		hostRef[i] = h_A[i] + h_B[i];
	}
}

void CheckResults(float* hostRef, float* gpuRef, size_t nElem)
{
	bool bSame = true;
	for (size_t i = 0; i < nElem; i++)
	{
		if (abs(gpuRef[i] - hostRef[i]) > 1e-5)
		{
			bSame = false;
		}
	}

	if (bSame)
	{
		printf("Result is correct!\n");
	}
	else
	{
		printf("Result is error!\n");
	}
}

__global__ void GpuSumArrays(float* d_A, float* d_B, float* d_C, size_t nElem)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nElem)
		d_C[tid] = d_A[tid] + d_B[tid];
}

int main()
{
	int nDev = 0;
	cudaSetDevice(nDev);

	cudaDeviceProp stDeviceProp;
	cudaGetDeviceProperties(&stDeviceProp, nDev);

	cudaGetDevice(&nDev);

	//check whether support mapped memory
	if (!stDeviceProp.canMapHostMemory)
	{
		printf("Device %d does not support mapping CPU host memory!\n", nDev);
		goto EXIT;
	}

	printf("Using device %d: %s\n", nDev, stDeviceProp.name);

	// set up data size of vector
	int nPower = 10;
	int nElem = 1 << nPower;
	size_t nBytes = nElem * sizeof(float);
	if (nPower < 18) {
		printf("Vector size %d power %d nbytes %3.0f KB\n",
			nElem, nPower, (float)nBytes / (1024.0f));
	}
	else {
		printf("Vector size %d power %d nbytes %3.0f MB\n",
			nElem, nPower, (float)nBytes / (1024.0f * 1024.0f));
	}

	// part2: using UMA - managed memory for array A , B, C
	// allocate UMA memory
	cudaError_t err = cudaMallocManaged(&h_A, nBytes);
	if (err != cudaSuccess)
	{
		printf("Not support cudaMallocManaged!\n");
		goto EXIT;
	}

	cudaMallocManaged(&h_B, nBytes, cudaMemAttachGlobal);
	cudaMallocManaged(&h_C, nBytes, cudaMemAttachGlobal);


	// initialize data at host side
	InitData(h_A, nElem);
	InitData(h_B, nElem);
	InitData(h_C, nElem);
	memset(hostRef, 0, nBytes);
	
	// add at host side for result checks
	SumArraysOnHost(h_A, h_B, hostRef, nElem);

	//execute kernle with zero copy memory
	GpuSumArrays << <grid, block >> > (h_A, h_B, h_C, nElem);

	// must be add the code before access the unified managed memory, 
	// otherwise will throw undefined exception
	cudaDeviceSynchronize();

	//check device results
	CheckResults(hostRef, h_C, nElem);

	// free memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	
	
	free(hostRef);
	free(gpuRef);

EXIT:
	cudaDeviceReset();

	system("pause");
	return 0;
}
