#include "cuda_launch_config.hpp"

int get_nearest_power(int val)
{
  return pow(2, ceil(log(val)/log(2)));
}

int get_lower_pow_neighbour(int x)
{ // http://my.safaribooksonline.com/book/information-technology-and-software-development/0201914654/power-of-2-boundaries/ch03lev1sec2
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    int res = x - (x >> 1);
    res = res >> 1;

    return (res<32)?32:res;
}

size_t use_cuda_launch_config(int regsPerBlock, int threadsPerBlock)
{
  return block_size_with_maximum_potential_occupancy(regsPerBlock, get_lower_pow_neighbour(threadsPerBlock));
}
