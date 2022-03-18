#include<string.h>
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#define N 100

//GPU VectorAddition Function
__global__ void vecAdd(float* A,float* B,float* C){
    int i=threadIdx.x;
    if(i<N)
        C[i]=A[i]+B[i];
}

int main(){
    //Some initializations of the elements.
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C, *h_D;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_D = (float*)malloc(size);
    float* d_A;
    cudaMalloc((void**)&d_A, size);
    float* d_B;
    cudaMalloc((void**)&d_B, size);
    float* d_C;
    cudaMalloc((void**)&d_C, size);
    srand(time(NULL));
    for(int i=0;i<N;i++){
        h_A[i] = rand()%100;
        h_B[i] = rand()%100;
    }
  
    //Get the CPU running time, using cudaEvent method
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i=0;i<N;i++){
        h_D[i] = h_A[i] + h_B[i];
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Processing time: %f (ms)\n", elapsedTime);
  
    //Use the GPU to check whether the addition is right.
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int threadsPerGrid = (N + threadsPerBlock-1)/threadsPerBlock;
    vecAdd<<<threadsPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  
    //Check the result of the vector addition.
    for(int i = 0; i < N; i++){
        if(h_C[i] = h_D[i]){
            ;
        }
        else{
            printf("Erro! The vector addition is wrong!\n");
            return -1;
        }
    }
    printf("The vector addition is right!\n");
  
    //Free the space of the vector.
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
