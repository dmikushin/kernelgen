#include "cuda_launch_config.hpp"

__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}

//template<typename T>
std::size_t use_cuda_launch_config(int regsPerBlock, int threadsPerBlock)
{
  std::cout << std::endl << "****CLC**** IN" << std::endl << std::flush;
  std::size_t block_size =  block_size_with_maximum_potential_occupancy(regsPerBlock, threadsPerBlock);
  std::cout << std::endl << "****CLC**** OUT(" << block_size << ")" << std::endl << std::flush;
  return block_size;
}
