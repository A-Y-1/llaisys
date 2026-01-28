#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float g, u;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            g = llaisys::utils::cast<float>(gate[i]);
            u = llaisys::utils::cast<float>(up[i]);
        } else {
            g = gate[i];
            u = up[i];
        }
        float tmp = u * g * (1.0f / (1.0f + std::exp(-g)));
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(tmp);
        } else {
            out[i] = tmp;
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                    reinterpret_cast<const llaisys::bf16_t *>(up), size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                    reinterpret_cast<const llaisys::fp16_t *>(up), size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
