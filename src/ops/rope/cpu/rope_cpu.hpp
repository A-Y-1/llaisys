#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_idx, float theta, llaisysDataType_t type, const std::vector<size_t> &shape);
}