#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_idx, float theta, const std::vector<size_t> &shape) {
    size_t seqlen = shape[0];
    size_t nhead = shape[1];
    size_t dims = shape[2];;

    std::vector<double> inv_freq(dims / 2);
    for (size_t k = 0; k < dims / 2; ++k) {
        inv_freq[k] = std::pow(theta, -2.0f * (static_cast<double>(k) / dims));
    }

    for(size_t i = 0; i < seqlen; i++) {
        double pos = pos_idx[i];
        std::vector<double> row_cos(dims / 2);
        std::vector<double> row_sin(dims / 2);
        for (size_t k = 0; k < dims / 2; ++k) {
            double phi = pos * inv_freq[k];
            row_cos[k] = std::cos(phi);
            row_sin[k] = std::sin(phi);
        }

        for(size_t j = 0; j < nhead; j++) {
            for(size_t k = 0; k < dims / 2; k++) {
                if constexpr( std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                    float ain = llaisys::utils::cast<float>(in[i * nhead * dims + j * dims + k]);
                    float bin = llaisys::utils::cast<float>(in[i * nhead * dims + j * dims + k + dims / 2]);
                    float aout = ain * row_cos[k] - bin * row_sin[k];
                    float bout = ain * row_sin[k] + bin * row_cos[k];
                    out[i * nhead * dims + j * dims + k] = llaisys::utils::cast<T>(aout);
                    out[i * nhead * dims + j * dims + k + dims / 2] = llaisys::utils::cast<T>(bout);
                }else{
                    double ain = in[i * nhead * dims + j * dims + k];
                    double bin = in[i * nhead * dims + j * dims + k + dims / 2];
                    out[i * nhead * dims + j * dims + k] = ain * row_cos[k] - bin * row_sin[k];
                    out[i * nhead * dims + j * dims + k + dims / 2] = ain * row_sin[k] + bin * row_cos[k];
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_idx, float theta, llaisysDataType_t type, const std::vector<size_t> &shape) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_idx), theta, shape);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_idx), theta, shape);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_idx), theta, shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
