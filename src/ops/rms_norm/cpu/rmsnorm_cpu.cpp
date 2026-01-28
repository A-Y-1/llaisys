#include "rmsnorm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t dim0, size_t dim1) {
    if constexpr(std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) { 
        for(size_t i = 0; i < dim0; i++){
            float sum_square = 0.0f;
            for(size_t j = 0; j < dim1; j++){
                sum_square += llaisys::utils::cast<float>(in[i * dim1 + j]) * llaisys::utils::cast<float>(in[i * dim1 + j]);
            }
            float rrms = 1.0f  / std::sqrt(sum_square / dim1 + eps);
            for(size_t j = 0; j < dim1; j++){
                out[i * dim1 + j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in[i * dim1 + j]) * rrms * llaisys::utils::cast<float>(weight[j]));
            }
        }
    }else{
        for(size_t i = 0; i < dim0; i++){
            float sum_square = 0.0f;
            for(size_t j = 0; j < dim1; j++){
                sum_square += in[i * dim1 + j] * in[i * dim1 + j];
            }
            float rrms = 1.0f  / std::sqrt(sum_square / dim1 + eps);
            for(size_t j = 0; j < dim1; j++){
                out[i * dim1 + j] = in[i * dim1 + j] * rrms * weight[j];
            }
        }
    }

}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t dim0, size_t dim1){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, dim0, dim1);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), eps, dim0, dim1);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), eps, dim0, dim1);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
