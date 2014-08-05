
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <iostream>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>

#define REPETITIONS	1000

#define NS_PER_SECOND	1000000000L


#define VECTOR_SIZE	1000*1024
// CUDA kernel. Each thread takes care of one element of c
// From https://www.olcf.ornl.gov/tutorials/cuda-vector-addition
__global__ 
void kernel_vector_sum(float *dst, float *v, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        dst[id] += v[id];
}


void cpu_vector_sum(float *dst, float *v, int n)
{
  for (int i=0; i<n; i++) 
        dst[n] += v[n];
}


void vectorSumTest()
{
  struct timespec t0, t1; 
  uint64_t dt;
  float *dst_cpu = new float[VECTOR_SIZE];
  float *v_cpu = new float[VECTOR_SIZE];
  std::cout << "CPU Vector Sum test. ";
  dt = 0;
  for (int r=0; r<REPETITIONS; r++) 
  {
    clock_gettime(CLOCK_MONOTONIC, &t0); 
    cpu_vector_sum(dst_cpu, v_cpu, VECTOR_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &t1); 
    dt += NS_PER_SECOND * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  }
  std::cout << "Time: " << ((double)dt/(double)(REPETITIONS*NS_PER_SECOND)) << " s (" << (dt/REPETITIONS) << " ns)" << std::endl;

  float *dst_gpu;
  float *v_gpu;
  int numCudaThreads = 1;
  cudaMalloc(&dst_gpu, VECTOR_SIZE*sizeof(float));
  cudaMalloc(&v_gpu, VECTOR_SIZE*sizeof(float));
  dt = 0;
  int blockSize = 1024;
  int gridSize = (int)ceil((float)numCudaThreads/blockSize);
  std::cout << "GPU Vector Sum test. ";
  for (int r=0; r<REPETITIONS; r++) 
  {
    clock_gettime(CLOCK_MONOTONIC, &t0); 
    kernel_vector_sum <<< gridSize, blockSize >>> (dst_gpu, v_gpu, VECTOR_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &t1);     
    dt += NS_PER_SECOND * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  }
  std::cout << "Time for "<< numCudaThreads << " threads: " << ( (double)dt/(double)(REPETITIONS*NS_PER_SECOND)) << " s (" << (dt/REPETITIONS) << " ns)" << std::endl;
  
  
  numCudaThreads = 64;
  dt = 0;
  blockSize = 1024;
  gridSize = (int)ceil((float)numCudaThreads/blockSize);
  std::cout << "GPU Vector Sum test. ";
  for (int r=0; r<REPETITIONS; r++) 
  {
    clock_gettime(CLOCK_MONOTONIC, &t0); 
    kernel_vector_sum <<< gridSize, blockSize >>> (dst_gpu, v_gpu, VECTOR_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &t1);     
    dt += NS_PER_SECOND * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  }
  std::cout << "Time for "<< numCudaThreads << " threads: " << ( (double)dt/(double)(REPETITIONS*NS_PER_SECOND)) << " s (" << (dt/REPETITIONS) << " ns)" << std::endl;

  numCudaThreads = 128;
  dt = 0;
  blockSize = 1024;
  gridSize = (int)ceil((float)numCudaThreads/blockSize);
  std::cout << "GPU Vector Sum test. ";
  for (int r=0; r<REPETITIONS; r++) 
  {
    clock_gettime(CLOCK_MONOTONIC, &t0); 
    kernel_vector_sum <<< gridSize, blockSize >>> (dst_gpu, v_gpu, VECTOR_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &t1);     
    dt += NS_PER_SECOND * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  }
  std::cout << "Time for "<< numCudaThreads << " threads: " << ( (double)dt/(double)(REPETITIONS*NS_PER_SECOND)) << " s (" << (dt/REPETITIONS) << " ns)" << std::endl;
  
  numCudaThreads = 128;
  dt = 0;
  blockSize = 1024;
  gridSize = (int)ceil((float)numCudaThreads/blockSize);
  std::cout << "GPU Vector Sum test. ";
  const int vsize = VECTOR_SIZE*sizeof(float);
  for (int r=0; r<REPETITIONS; r++) 
  {
    clock_gettime(CLOCK_MONOTONIC, &t0); 
    cudaMemcpy( dst_gpu, dst_cpu, vsize, cudaMemcpyHostToDevice );
    cudaMemcpy( v_gpu, v_cpu, vsize, cudaMemcpyHostToDevice );   
    kernel_vector_sum <<< gridSize, blockSize >>> (dst_gpu, v_gpu, VECTOR_SIZE);
    cudaMemcpy( dst_cpu, dst_gpu, vsize, cudaMemcpyDeviceToHost );
    clock_gettime(CLOCK_MONOTONIC, &t1);     
    dt += NS_PER_SECOND * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  }
  std::cout << "Time for "<< numCudaThreads << " threads (with memcpy): " << ( (double)dt/(double)(REPETITIONS*NS_PER_SECOND)) << " s (" << (dt/REPETITIONS) << " ns)" << std::endl;

  cudaFree(&dst_gpu);
  cudaFree(&v_gpu);
  delete[] v_cpu;
  delete[] dst_cpu;
  
}



#define LOOP_COUNT	10000000


__global__
void gpu_loopTest()
{
  long tmp=0;
  for (int i=0; i<LOOP_COUNT; i++) {
    tmp++;
  }
}

void cpu_loopTest()
{
  long tmp=0;
  for (int i=0; i<LOOP_COUNT; i++) {
    tmp++;
  }
}

// Loop
void loopTest()
{
  struct timespec t0, t1; 
  uint64_t dt;
  std::cout << "CPU Loop test. ";
  dt = 0;
  for (int r=0; r<REPETITIONS; r++) 
  {
    clock_gettime(CLOCK_MONOTONIC, &t0); 
    cpu_loopTest();
    clock_gettime(CLOCK_MONOTONIC, &t1); 
    dt += NS_PER_SECOND * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  }
  std::cout << "Time: " << ((double)dt/(double)(REPETITIONS*NS_PER_SECOND)) << " s (" << (dt/REPETITIONS) << " ns)" << std::endl;

  dt = 0;
  dim3 dimBlock( 1, 1 );
  dim3 dimGrid( 1, 1 );
  std::cout << "GPU Loop test. ";
  for (int r=0; r<REPETITIONS; r++) 
  {
    clock_gettime(CLOCK_MONOTONIC, &t0); 
    gpu_loopTest <<< dimGrid, dimBlock >>> ();
    clock_gettime(CLOCK_MONOTONIC, &t1);     
    dt += NS_PER_SECOND * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  }
  std::cout << "Time: " << ( (double)dt/(double)(REPETITIONS*NS_PER_SECOND)) << " s (" << (dt/REPETITIONS) << " ns)" << std::endl;
}


int main() 
{
  std::cout.imbue( std::locale("") );
  std::cout << "CUDA test" << std::endl;
  std::cout << "Averages for " << REPETITIONS << " repetitions." << std::endl;
  
  loopTest();
  vectorSumTest();

  return 0;
}
