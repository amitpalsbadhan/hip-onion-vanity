#pragma once
#include <cstdint>
#include <cstddef>

namespace sha3 {
    void sha3_256(const uint8_t *in, size_t inlen, uint8_t *md);
}
