#include "cuda_launch_config.hpp"

size_t use_cuda_launch_config(int regsPerBlock, int threadsPerBlock)
{
  return block_size_with_maximum_potential_occupancy(regsPerBlock, threadsPerBlock);
}
