#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
 
int main()
{
	size_t avail;
	size_t total;
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);              // 用deviceCount获取显卡总数量
	for(int i_dev = 0;i_dev < deviceCount;i_dev++)
	{
		cudaSetDevice(i_dev);		               // 第i_dev张显卡
		cudaMemGetInfo(&avail, &total);            // 获取可用和总显存大小
		cudaDeviceProp dveProp;
		cudaGetDeviceProperties(&dveProp, i_dev);
		printf("Device %d Information:\n",i_dev);
		printf("显卡设备%d:%s\n", i_dev, dveProp.name);
		printf("Avaliable Memery = %dm   Total Memory = %dm\n", int(avail/1024/1024), int(total / 1024 / 1024));
		printf("SM数量:%d\n", dveProp.multiProcessorCount);
		printf("每个SM中共享内存的大小:%lfKB\n", dveProp.sharedMemPerMultiprocessor / 1024.0f);
		printf("共享内存大小:%lfKB\n", dveProp.multiProcessorCount * dveProp.sharedMemPerMultiprocessor / 1024.0f);
		printf("==================================================================\n");	
	}	
    return 0;
}
