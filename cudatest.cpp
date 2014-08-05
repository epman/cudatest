#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#include <iostream>

int main() 
{
  
  std::cout << "CUDA test" << std::endl;
  struct timespec t0, t1; 
  clock_gettime(CLOCK_MONOTONIC, &t0); 
  long tmp=0;
  for (int i=0; i<100; i++) {
    tmp++;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1); 
  uint64_t dt = 1000000000L * (t1.tv_sec - t0.tv_sec) + t1.tv_nsec - t0.tv_nsec;
  std::cout << "Time: " << dt << " ns" << std::endl;
  return 0;
}
