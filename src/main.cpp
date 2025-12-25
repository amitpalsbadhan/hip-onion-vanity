#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <csignal>
#include <atomic>
#include <thread>
#include <random>

#include "config.h"
#include "kernels.h"
#include "utils.h"

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }

// Helper: Add a 64-bit integer to a 256-bit struct (CPU side)
void host_u256_add(u256& val, uint64_t add) {
    uint64_t c = add;
    uint64_t sum = (uint64_t)val.v[0] + c;
    val.v[0] = (uint32_t)sum;
    c = sum >> 32;
    for(int i=1; i<8 && c; i++) {
        sum = (uint64_t)val.v[i] + c;
        val.v[i] = (uint32_t)sum;
        c = sum >> 32;
    }
}

// Helper: Generate a fresh 256-bit random key from System Entropy
void generate_secure_base(u256& key) {
    // std::random_device asks the OS for entropy (e.g., /dev/urandom)
    std::random_device rd; 
    for(int i=0; i<8; i++) {
        key.v[i] = rd(); 
    }
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    std::vector<std::string> target_prefixes;
    int timeout = 0;
    int limit = 0; 
    int blocks_per_grid = DEFAULT_BLOCKS_PER_GRID;
    int batches_per_kernel = DEFAULT_BATCHES_PER_KERNEL;
    
    for(int i=1; i<argc; i++) {
        std::string arg = argv[i];
        if(arg == "-p" && i+1 < argc) target_prefixes.push_back(argv[++i]);
        if(arg == "-t" && i+1 < argc) timeout = std::stoi(argv[++i]);
        if(arg == "-n" && i+1 < argc) limit = std::stoi(argv[++i]);
        if(arg == "-b" && i+1 < argc) blocks_per_grid = std::stoi(argv[++i]);
        if(arg == "-l" && i+1 < argc) batches_per_kernel = std::stoi(argv[++i]);
    }

    if(target_prefixes.empty()) {
        std::cout << "Usage: " << argv[0] << " -p <prefix> [-n <count>] [-t <seconds>] [-b <blocks>] [-l <batches>]" << std::endl;
        return 1;
    }

    gpu_prefixes_t host_prefixes;
    host_prefixes.count = 0;
    for(const auto& s : target_prefixes) {
        if(host_prefixes.count >= MAX_PREFIXES) break;
        if(s.length() > MAX_PREFIX_LEN) continue;
        host_prefixes.lengths[host_prefixes.count] = s.length();
        for(size_t j=0; j<s.length(); j++) {
            host_prefixes.data[host_prefixes.count][j] = get_b32_index(s[j]);
        }
        host_prefixes.count++;
    }

    int deviceId;
    hipGetDevice(&deviceId);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, deviceId);
    std::cout << "GPU: " << props.name << " (Batch Optimized)" << std::endl;
    std::cout << "Config: " << blocks_per_grid << " Blocks, " << batches_per_kernel << " Batches of " << BATCH_SIZE << " keys." << std::endl;

    search_result_t host_res = {0};
    search_result_t* dev_res;
    hipMalloc(&dev_res, sizeof(search_result_t));
    hipMemset(dev_res, 0, sizeof(search_result_t));

    // Generate 256-bit Random Base
    u256 base_key;
    std::cout << "[*] Seeding from System Entropy..." << std::endl;
    generate_secure_base(base_key);

    uint64_t batch_size = (uint64_t)blocks_per_grid * THREADS_PER_BLOCK * batches_per_kernel * BATCH_SIZE;
    uint64_t total_checked = 0;
    int matches_found = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print = start_time;

    while(running) {
        launch_vanity_search(blocks_per_grid, THREADS_PER_BLOCK, dev_res, base_key, host_prefixes, batches_per_kernel);
        hipDeviceSynchronize();
        
        hipMemcpy(&host_res, dev_res, sizeof(search_result_t), hipMemcpyDeviceToHost);
        
        if(host_res.found) {
            save_result(host_res, target_prefixes);
            matches_found++;

            // RE-RANDOMIZE BASE KEY AFTER MATCH
            // This ensures the next key found is mathematically unrelated
            // to the one we just found.
            std::cout << "    -> Re-seeding base key for unlinkability..." << std::endl;
            generate_secure_base(base_key);

            hipMemset(dev_res, 0, sizeof(search_result_t));
            if (limit > 0 && matches_found >= limit) break;
        } else {
            // Only increment linearly if we DIDN'T find a match.
            host_u256_add(base_key, batch_size);
        }

        total_checked += batch_size;

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;
        std::chrono::duration<double> since_print = now - last_print;
        
        if(timeout > 0 && elapsed.count() > timeout) break;

        if(since_print.count() > 1.0) {
            double mkeys = (double)total_checked / elapsed.count() / 1000000.0;
            std::cout << "\rSpeed: " << std::fixed << std::setprecision(2) << mkeys << " Mops/s | Found: " << matches_found << " | Total: " << (total_checked/1000000) << "M" << std::flush;
            last_print = now;
        }
    }

    hipFree(dev_res);
    return 0;
}
