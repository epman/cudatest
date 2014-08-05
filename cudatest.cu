
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#include <iostream>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>

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


int main() 
{
  
  std::cout << "CUDA test" << std::endl;
  struct timespec t0, t1; 

  // Loop
  clock_gettime(CLOCK_MONOTONIC, &t0); 
  cpu_loopTest();
  clock_gettime(CLOCK_MONOTONIC, &t1); 
  uint64_t dt = 1000000000L * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  std::cout << "CPU Loop test. Time: " << dt << " ns" << std::endl;

  clock_gettime(CLOCK_MONOTONIC, &t0); 
  dim3 dimBlock( 1, 1 );
  dim3 dimGrid( 1, 1 );
  gpu_loopTest <<< dimGrid, dimBlock >>> ();
  clock_gettime(CLOCK_MONOTONIC, &t1); 
  dt = 1000000000L * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  std::cout << "GPU Loop test. Time: " << dt << " ns" << std::endl;

  return 0;
}
