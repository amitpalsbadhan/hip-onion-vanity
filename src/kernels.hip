#include <hip/hip_runtime.h>
#include "kernels.h"

// GPU MATH (RDNA 2 Optimized)

typedef struct { uint32_t v[8]; } fe;
typedef struct { fe X; fe Z; } fe_p1;

__device__ const uint32_t P[8] = {
    0xFFFFFFED, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF
};

__device__ __forceinline__ void fe_add(fe *out, const fe *a, const fe *b) {
    uint32_t t[8]; uint64_t c = 0;
    #pragma unroll
    for(int i=0; i<8; i++) { uint64_t s = (uint64_t)a->v[i] + b->v[i] + c; t[i] = (uint32_t)s; c = s >> 32; }
    uint32_t s[8]; uint64_t borrow = 0;
    #pragma unroll
    for(int i=0; i<8; i++) { uint64_t d = (uint64_t)t[i] - P[i] - borrow; s[i] = (uint32_t)d; borrow = (d >> 63) & 1; }
    uint32_t mask = (borrow == 0) ? 0xFFFFFFFF : 0;
    uint32_t not_mask = ~mask;
    #pragma unroll
    for(int i=0; i<8; i++) { out->v[i] = (s[i] & mask) | (t[i] & not_mask); }
}

__device__ __forceinline__ void fe_sub(fe *out, const fe *a, const fe *b) {
    uint32_t t[8]; uint64_t borrow = 0;
    #pragma unroll
    for(int i=0; i<8; i++) { uint64_t d = (uint64_t)a->v[i] - b->v[i] - borrow; t[i] = (uint32_t)d; borrow = (d >> 63) & 1; }
    uint32_t mask = (borrow) ? 0xFFFFFFFF : 0; uint64_t c = 0;
    #pragma unroll
    for(int i=0; i<8; i++) { uint32_t p = P[i] & mask; uint64_t s = (uint64_t)t[i] + p + c; out->v[i] = (uint32_t)s; c = s >> 32; }
}

__device__ __forceinline__ void fe_mul(fe *out, const fe *a, const fe *b) {
    uint32_t t[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) { uint64_t p = (uint64_t)a->v[i] * b->v[j] + t[i+j] + carry; t[i+j] = (uint32_t)p; carry = p >> 32; }
        t[i+8] += (uint32_t)carry;
    }
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { uint64_t r = (uint64_t)t[i+8] * 38 + t[i] + carry; out->v[i] = (uint32_t)r; carry = r >> 32; }
    uint64_t r = (uint64_t)out->v[0] + (carry * 19); out->v[0] = (uint32_t)r; carry = r >> 32;
    #pragma unroll
    for(int i=1; i<8; i++) { uint64_t s = (uint64_t)out->v[i] + carry; out->v[i] = (uint32_t)s; carry = s >> 32; }
}

__device__ __forceinline__ void fe_sq(fe *out, const fe *a) { fe_mul(out, a, a); }

__device__ __forceinline__ void fe_cswap(fe *a, fe *b, int swap) {
    uint32_t mask = swap ? 0xFFFFFFFF : 0;
    #pragma unroll
    for(int i=0; i<8; i++) { uint32_t x = mask & (a->v[i] ^ b->v[i]); a->v[i] ^= x; b->v[i] ^= x; }
}

__device__ __forceinline__ void fe_invert(fe *out, const fe *z) {
    fe t0, t1, t2, t3;
    fe_sq(&t0, z); fe_sq(&t1, &t0); fe_sq(&t1, &t1); fe_mul(&t1, z, &t1); fe_mul(&t0, &t0, &t1);
    fe_sq(&t2, &t0); fe_mul(&t1, &t1, &t2); fe_sq(&t2, &t1);
    for (int i = 1; i < 5; ++i) fe_sq(&t2, &t2); fe_mul(&t1, &t2, &t1); fe_sq(&t2, &t1);
    for (int i = 1; i < 10; ++i) fe_sq(&t2, &t2); fe_mul(&t2, &t2, &t1); fe_sq(&t3, &t2);
    for (int i = 1; i < 20; ++i) fe_sq(&t3, &t3); fe_mul(&t2, &t3, &t2); fe_sq(&t2, &t2);
    for (int i = 1; i < 10; ++i) fe_sq(&t2, &t2); fe_mul(&t1, &t2, &t1); fe_sq(&t2, &t1);
    for (int i = 1; i < 50; ++i) fe_sq(&t2, &t2); fe_mul(&t2, &t2, &t1); fe_sq(&t3, &t2);
    for (int i = 1; i < 100; ++i) fe_sq(&t3, &t3); fe_mul(&t2, &t3, &t2); fe_sq(&t2, &t2);
    for (int i = 1; i < 50; ++i) fe_sq(&t2, &t2); fe_mul(&t1, &t2, &t1); fe_sq(&t1, &t1);
    for (int i = 1; i < 5; ++i) fe_sq(&t1, &t1); fe_mul(out, &t1, &t0);
}

__device__ __forceinline__ void fe_batch_invert(fe *out, const fe *in, int n) {
    fe products[BATCH_SIZE];
    products[0] = in[0];
    for(int i=1; i<n; i++) fe_mul(&products[i], &products[i-1], &in[i]);
    fe total_inv;
    fe_invert(&total_inv, &products[n-1]);
    for(int i=n-1; i>0; i--) {
        fe_mul(&out[i], &products[i-1], &total_inv);
        fe_mul(&total_inv, &total_inv, &in[i]);
    }
    out[0] = total_inv;
}

__device__ __forceinline__ void differential_add(fe_p1 *P_next, const fe_p1 *P, const fe_p1 *Q, const fe_p1 *P_prev) {
    fe t1, t2, t3, t4;
    fe_add(&t1, &P->X, &P->Z); fe_sub(&t2, &P->X, &P->Z);
    fe t_base_sum = {{10,0,0,0,0,0,0,0}}; fe t_base_diff = {{8,0,0,0,0,0,0,0}};
    fe_mul(&t3, &t1, &t_base_diff); fe_mul(&t4, &t2, &t_base_sum);
    fe_add(&t1, &t3, &t4); fe_sub(&t2, &t3, &t4);
    fe_sq(&t1, &t1); fe_sq(&t2, &t2);
    fe_mul(&P_next->X, &P_prev->Z, &t1); fe_mul(&P_next->Z, &P_prev->X, &t2);
}

__device__ __forceinline__ void montgomery_ladder_init(fe_p1 *P, const u256 *scalar) {
    fe_p1 x0, x1;
    for(int i=0; i<8; i++) { x0.X.v[i]=0; x0.Z.v[i]=0; } x0.X.v[0]=1; 
    for(int i=0; i<8; i++) { x1.X.v[i]=0; x1.Z.v[i]=0; } x1.X.v[0]=9; x1.Z.v[0]=1;
    fe_p1 G_base = x1; int swap = 0;
    for(int i = 254; i >= 0; --i) {
        int bit = (scalar->v[i / 32] >> (i % 32)) & 1;
        swap ^= bit;
        fe_cswap(&x0.X, &x1.X, swap); fe_cswap(&x0.Z, &x1.Z, swap);
        differential_add(&x1, &x1, &G_base, &x0); 
        fe t1, t2, t3, t4;
        fe_add(&t1, &x0.X, &x0.Z); fe_sq(&t1, &t1); 
        fe_sub(&t2, &x0.X, &x0.Z); fe_sq(&t2, &t2); 
        fe_mul(&x0.X, &t1, &t2); fe_sub(&t3, &t1, &t2); 
        fe c121666 = {{0x1DB42,0,0,0,0,0,0,0}};
        fe_mul(&t4, &t3, &c121666); fe_add(&t4, &t4, &t2);
        fe_mul(&x0.Z, &t3, &t4);
        swap = bit;
    }
    fe_cswap(&x0.X, &x1.X, swap); fe_cswap(&x0.Z, &x1.Z, swap);
    *P = x0;
}

__device__ __forceinline__ int check_prefixes(const fe *Y, const gpu_prefixes_t *prefixes) {
    uint8_t y_bytes[32];
    #pragma unroll
    for(int i=0; i<8; i++) {
        y_bytes[i*4+0] = Y->v[i] & 0xFF;
        y_bytes[i*4+1] = (Y->v[i] >> 8) & 0xFF;
        y_bytes[i*4+2] = (Y->v[i] >> 16) & 0xFF;
        y_bytes[i*4+3] = (Y->v[i] >> 24) & 0xFF;
    }

    for(int p=0; p < prefixes->count; p++) {
        bool match = true;
        int len = prefixes->lengths[p];
        
        uint64_t buffer = 0;
        int bits_left = 0;
        int byte_idx = 0;

        for(int i=0; i<len; i++) {
            while (bits_left < 5 && byte_idx < 32) {
                buffer = (buffer << 8) | y_bytes[byte_idx++];
                bits_left += 8;
            }
            uint8_t chunk = (buffer >> (bits_left - 5)) & 0x1F;
            bits_left -= 5;

            if (chunk != prefixes->data[p][i]) {
                match = false;
                break;
            }
        }
        if (match) return p;
    }
    return -1;
}

// helper for 256-bit addition
__device__ __forceinline__ void u256_add_u64(u256 *val, uint64_t add) {
    uint64_t c = add;
    uint64_t sum = (uint64_t)val->v[0] + c;
    val->v[0] = (uint32_t)sum;
    c = sum >> 32;
    // Propagate carry up the chain
    for(int i=1; i<8 && c; i++) {
        sum = (uint64_t)val->v[i] + c;
        val->v[i] = (uint32_t)sum;
        c = sum >> 32;
    }
}

__global__ void vanity_kernel_batch(search_result_t* result, u256 base_key, gpu_prefixes_t prefixes, int batches_per_kernel) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_base_offset = tid * (BATCH_SIZE * batches_per_kernel);
    
    // Initialize scalar with the Random Base Key
    u256 scalar = base_key;

    // Add the thread's unique offset to the random base
    u256_add_u64(&scalar, thread_base_offset);

    // Apply Clamping (Standard Ed25519 security)
    scalar.v[0] &= 0xFFFFFFF8; 
    scalar.v[7] &= 0x7FFFFFFF; 
    scalar.v[7] |= 0x40000000; 

    fe_p1 P_prev, P_curr, P_next;
    montgomery_ladder_init(&P_prev, &scalar);
    
    fe_p1 G;
    G.X.v[0] = 9; for(int i=1; i<8; i++) G.X.v[i] = 0;
    G.Z.v[0] = 1; for(int i=1; i<8; i++) G.Z.v[i] = 0;
    
    differential_add(&P_curr, &P_prev, &G, &P_prev);

    fe nums[BATCH_SIZE];
    fe dens[BATCH_SIZE];
    fe inv_dens[BATCH_SIZE];

    for (int b = 0; b < batches_per_kernel; b++) {
        for(int i=0; i<BATCH_SIZE; i++) {
            fe_sub(&nums[i], &P_curr.X, &P_curr.Z);
            fe_add(&dens[i], &P_curr.X, &P_curr.Z);
            differential_add(&P_next, &P_curr, &G, &P_prev);
            P_prev = P_curr;
            P_curr = P_next;
        }

        fe_batch_invert(inv_dens, dens, BATCH_SIZE);

        for(int i=0; i<BATCH_SIZE; i++) {
            fe Y;
            fe_mul(&Y, &nums[i], &inv_dens[i]);

            int match_idx = check_prefixes(&Y, &prefixes);
            if (match_idx >= 0) {
                if (atomicCAS(&result->found, 0, 1) == 0) {
                    result->thread_id = tid;
                    result->batch_index = i;
                    result->prefix_index = match_idx;
                    result->public_key_y.v[0] = Y.v[0]; 
                    for(int k=1; k<8; k++) result->public_key_y.v[k] = Y.v[k];
                    
                    u256 win_key = scalar;
                    uint64_t add_val = (b * BATCH_SIZE) + i + 1;
                    uint64_t c = 0;
                    uint64_t sum = (uint64_t)win_key.v[0] + add_val;
                    win_key.v[0] = (uint32_t)sum;
                    c = sum >> 32;
                    for(int k=1; k<8 && c; k++) {
                        sum = (uint64_t)win_key.v[k] + c;
                        win_key.v[k] = (uint32_t)sum;
                        c = sum >> 32;
                    }
                    result->private_key = win_key;
                }
                return;
            }
        }
    }
}

// IMPLEMENTATION OF THE WRAPPER
void launch_vanity_search(int blocks, int threads, search_result_t* dev_res, u256 base_key, gpu_prefixes_t prefixes, int batches) {
    hipLaunchKernelGGL(vanity_kernel_batch, dim3(blocks), dim3(threads), 0, 0, dev_res, base_key, prefixes, batches);
}
