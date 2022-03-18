#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp dveProp;
        cudaGetDeviceProperties(&dveProp, i);
        printf("  --- General Information for device %d --- \n", i);
        printf("显卡设备%d:%s\n", i, dveProp.name);
        printf("计算能力: %d.%d\n", dveProp.major, dveProp.minor);
        printf("SM数量:%d\n", dveProp.multiProcessorCount);
        printf("时钟速率: %d\n", dveProp.clockRate);
        printf("设备是否能够同时进行cudaMemcpy()和内核执行:");
        if(dveProp.deviceOverlap)
            printf("Enable\n");
        else
            printf("Disable\n");

        printf("设备上执行的内核是否有运行时的限制:");
        if(dveProp.kernelExecTimeoutEnabled)
            puts("Enable");
        else
            puts("Disable");

        printf("  --- Memory Information for device %d ---\n", i);
        printf("全局内存大小:%lfMB\n", dveProp.totalGlobalMem / 1024.0f / 1024.0f);
        printf("常量内存大小: %ld\n", dveProp.totalConstMem);
        printf("内存中允许的最大间距字节数: %ld\n", dveProp.memPitch);
        printf("设备对纹理对齐的要求: %ld\n", dveProp.textureAlignment);

        printf("  --- MP Information for device %d --- \n", i);
        printf("设备上多处理器的数量:%d\n", dveProp.multiProcessorCount);
        printf("每个线程块的共享内存大小:%lfKB\n", dveProp.sharedMemPerBlock / 1024.0f);
        printf("每个线程块的最大线程数:%d\n", dveProp.maxThreadsPerBlock);
        printf("每个线程块的可用寄存器数量:%d\n", dveProp.regsPerBlock);
        printf("每个SM的最大线程数:%d\n", dveProp.maxThreadsPerMultiProcessor);
        printf("Thread Warp size: %d\n", dveProp.warpSize);
        printf("每个SM的最大线程束数:%d\n", dveProp.maxThreadsPerMultiProcessor / dveProp.warpSize);
        printf("Maximum sizes of each dimension of a block: (%d, %d, %d)\n", dveProp.maxThreadsDim[0], dveProp.maxThreadsDim[1], dveProp.maxThreadsDim[2]);
        printf("Maximum sizes of each dimension of a grid: (%d, %d, %d)\n", dveProp.maxGridSize[0], dveProp.maxGridSize[1], dveProp.maxGridSize[2]);
        printf("==================================================================\n");
    }
    return 0;
}
