#include "embedding_cpu.hpp"

#include "../../../utils.hpp"
#include "../../../core/llaisys_core.hpp"

#include <cmath>
#include <cstring>

// template <typename T>
// void embedding_(T *out, const int64_t *index, const T *weight, const int64_t width, size_t numel) {
//     size_t copy_size = width * sizeof(T);
//     for(size_t i = 0; i < numel; i++){
//         std::memcpy(out + i * width, weight + index[i] * width, copy_size);
//     }
// }

template<typename T>
void embedding_(T *out, const int64_t *index, const T *weight, const int64_t width, size_t numel) {
    size_t copy_size = width * sizeof(T);

    for(size_t i = 0; i < numel; i++){
        int64_t token_id = index[i];
        const T* src = weight + (token_id * width);
        T* dst = out + (i * width);
        std::memcpy(dst, src, copy_size);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, const int64_t width, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), width, numel);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), width, numel);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), width, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
