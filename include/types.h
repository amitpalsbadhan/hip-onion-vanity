#pragma once
#include <cstdint>
#include "config.h"

// 256-bit integer (8 * 32-bit words)
typedef struct { uint32_t v[8]; } u256;

// Result structure passed between GPU and Host
typedef struct {
    uint32_t found;
    uint32_t thread_id;
    uint32_t batch_index;
    uint32_t prefix_index;
    u256 public_key_y; 
    u256 private_key;
} search_result_t;

// Prefix storage for GPU constant memory
typedef struct {
    uint8_t data[MAX_PREFIXES][MAX_PREFIX_LEN];
    uint8_t lengths[MAX_PREFIXES];
    int count;
} gpu_prefixes_t;
