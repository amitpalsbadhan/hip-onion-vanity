#pragma once
#include <cstdint>
#include "types.h"

// Wrapper function to launch the kernel.
// Implemented in src/kernels.hip
void launch_vanity_search(
    int blocks, 
    int threads, 
    search_result_t* dev_res, 
    u256 base_key, 
    gpu_prefixes_t prefixes, 
    int batches
);
