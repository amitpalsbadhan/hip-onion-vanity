#pragma once
#include <vector>
#include <string>
#include "types.h"

std::string base32_encode(const std::vector<uint8_t>& data);
int get_b32_index(char c);
void save_result(const search_result_t& res, const std::vector<std::string>& prefix_strs);
