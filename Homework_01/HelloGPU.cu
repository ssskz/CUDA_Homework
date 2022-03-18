#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void helloGPU(void)
{
	printf("thread_no: %d  Hello GPU!\n",threadIdx.x);
}
int main()
{
	//hello from CPU
	printf("Hello CPU!\n");
	helloGPU <<<2, 10 >>> ();
	cudaDeviceReset();
	return 0;
}
