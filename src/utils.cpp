#include "utils.h"
#include "sha3.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

std::string base32_encode(const std::vector<uint8_t>& data) {
    static const char* alphabet = "abcdefghijklmnopqrstuvwxyz234567";
    std::string output;
    int buf = 0, bits = 0;
    for (uint8_t b : data) {
        buf = (buf << 8) | b;
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            output += alphabet[(buf >> bits) & 0x1F];
        }
    }
    if (bits > 0) {
        output += alphabet[(buf << (5 - bits)) & 0x1F];
    }
    return output;
}

int get_b32_index(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= '2' && c <= '7') return c - '2' + 26;
    return -1;
}

void save_result(const search_result_t& res, const std::vector<std::string>& prefix_strs) {
    uint8_t pub_bytes[32];
    for(int i=0; i<8; i++) {
        pub_bytes[i*4+0] = res.public_key_y.v[i] & 0xFF;
        pub_bytes[i*4+1] = (res.public_key_y.v[i] >> 8) & 0xFF;
        pub_bytes[i*4+2] = (res.public_key_y.v[i] >> 16) & 0xFF;
        pub_bytes[i*4+3] = (res.public_key_y.v[i] >> 24) & 0xFF;
    }

    std::string final_url = "";
    uint8_t final_pub[32];

    for(int sign = 0; sign < 2; sign++) {
        uint8_t candidate_pub[32];
        memcpy(candidate_pub, pub_bytes, 32);
        if(sign) candidate_pub[31] |= 0x80;
        else     candidate_pub[31] &= 0x7F;

        uint8_t checksum_data[15 + 32 + 1];
        memcpy(checksum_data, ".onion checksum", 15);
        memcpy(checksum_data + 15, candidate_pub, 32);
        checksum_data[15 + 32] = 0x03; 

        uint8_t hash[32];
        sha3::sha3_256(checksum_data, sizeof(checksum_data), hash);

        std::vector<uint8_t> onion_raw;
        onion_raw.insert(onion_raw.end(), candidate_pub, candidate_pub + 32);
        onion_raw.push_back(hash[0]); 
        onion_raw.push_back(hash[1]); 
        onion_raw.push_back(0x03);    

        std::string url = base32_encode(onion_raw) + ".onion";
        
        if (sign == 0) {
            final_url = url;
            memcpy(final_pub, candidate_pub, 32);
        }
    }

    std::string filename = "found.txt";
    if (res.prefix_index < prefix_strs.size()) {
        filename = prefix_strs[res.prefix_index] + ".txt";
    }

    std::ofstream file(filename, std::ios::app);
    file << "==================================================" << std::endl;
    file << "Onion URL:   " << final_url << std::endl;
    file << "Private Key: ";
    for(int i=0; i<8; i++) file << std::hex << std::setw(8) << std::setfill('0') << res.private_key.v[i];
    file << std::dec << std::endl;
    file << "Public Key:  ";
    for(int i=0; i<32; i++) file << std::hex << std::setw(2) << std::setfill('0') << (int)final_pub[i];
    file << std::dec << std::endl;
    file << "NOTE: Private Key is a RAW SCALAR (not a seed)." << std::endl;
    file << "==================================================" << std::endl;
    file.close();
    
    std::cout << "\n[+] MATCH FOUND: " << final_url << " (Saved to " << filename << ")" << std::endl;
}
