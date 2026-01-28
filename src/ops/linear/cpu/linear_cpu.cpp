#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t K, size_t N, size_t ndim_bias) {
    if(bias == nullptr){
        for(size_t i = 0; i < M; i++){
            for(size_t j = 0; j < N; j++){
                float tmp = 0.0;
                for(size_t k = 0; k < K; k++){
                    if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                        tmp += llaisys::utils::cast<float>(in[i * K + k]) * llaisys::utils::cast<float>(weight[j * K + k]);
                    }else{
                        tmp += in[i * K + k] * weight[j * K + k];
                    }
                }
                if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                    out[i * N + j] = llaisys::utils::cast<T>(tmp);
                }else out[i * N + j] = tmp;
            }
        }
    }else{
        for(size_t i = 0; i < M; i++){
            for(size_t j = 0; j < N; j++){
                float tmp = 0.0;
                if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                    tmp = ndim_bias == 0 ? llaisys::utils::cast<float>(bias[0]) : llaisys::utils::cast<float>(bias[j]);
                }else{
                    tmp = 0 ? bias[0] : bias[j];
                }
                for(size_t k = 0; k < K; k++){
                    if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                        tmp += llaisys::utils::cast<float>(in[i * K + k]) * llaisys::utils::cast<float>(weight[j * K + k]);
                    }else{
                        tmp += in[i * K + k] * weight[j * K + k];
                    }
                }
                if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                    out[i * N + j] = llaisys::utils::cast<T>(tmp);
                }else out[i * N + j] = tmp;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t M, size_t K, size_t N, llaisysDataType_t type, size_t ndim_bias) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias), M, K, N, ndim_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), 
        reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), M, K, N, ndim_bias);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), M, K, N, ndim_bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
