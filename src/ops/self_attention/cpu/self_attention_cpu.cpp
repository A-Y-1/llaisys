#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cfloat>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, 
    const std::vector<size_t> &q_shape, const std::vector<size_t> &k_shape, const std::vector<size_t> &v_shape){ 
    size_t seqlen = q_shape[0];
    size_t nhead = q_shape[1];
    size_t dim = q_shape[2];
    size_t total_len = k_shape[0];
    size_t nkvhead = k_shape[1];
    size_t dimv = v_shape[2];

    int64_t past_len = static_cast<int64_t>(total_len) - static_cast<int64_t>(seqlen);
    if (past_len < 0) {
        past_len = 0;
    }

    std::vector<float> scores(total_len);
    for(size_t i = 0; i < seqlen; i++){
        for(size_t h = 0; h < nhead; h++){

            size_t kvh = h / (nhead / nkvhead);
            size_t mask_pos = (past_len + i) >= total_len ? (total_len - 1) : past_len + i;

            //attn_score
            for(size_t j = 0; j <= mask_pos; j++){
                float sum = 0.0f;
                for(size_t l = 0; l < dim; l++){
                    if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                        float qval = llaisys::utils::cast<float>(q[i * nhead * dim + h * dim + l]);
                        float kval = llaisys::utils::cast<float>(k[j * nkvhead * dim + kvh * dim + l]);
                        sum += qval * kval;
                    }else{
                        sum += q[i * nhead * dim + h * dim + l] * k[j * nkvhead * dim + kvh * dim + l];
                    }
                }
                scores[j] = sum * scale;
            }

            //softmax
            float max_score = -FLT_MAX;
            float softmax_sum = 0.0f;
            for(size_t j = 0; j <= mask_pos; j++){
                max_score = std::max(max_score, scores[j]);
            }
            for(size_t j = 0; j <= mask_pos; j++){
                scores[j] = std::exp(scores[j] - max_score);
                softmax_sum += scores[j];
            }

            //Y = casualsoftmax(A) * V
            float softmax_sum_inv = softmax_sum != 0.0 ? (1.0 / softmax_sum) : 1.0;
            for(size_t j = 0; j < dimv; j++){
                float sum = 0.0f;
                for(size_t l = 0; l <= mask_pos; l++){
                    if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                        float aval = llaisys::utils::cast<float>(scores[l]);
                        float vval = llaisys::utils::cast<float>(v[l * nkvhead * dimv + kvh * dimv + j]);
                        sum += aval * softmax_sum_inv * vval;
                    }else{
                        sum += scores[l] * softmax_sum_inv * v[l * nkvhead * dimv + kvh * dimv + j]; 
                    }
                }
                if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
                    attn_val[i * nhead * dimv + h * dimv + j] = llaisys::utils::cast<T>(sum);
                }else{
                    attn_val[i * nhead * dimv + h * dimv + j] = sum;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
    llaisysDataType_t type, float scale, const std::vector<size_t> &q_shape, const std::vector<size_t> &k_shape, const std::vector<size_t> &v_shape){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k),
                reinterpret_cast<const float *>(v), scale, q_shape, k_shape, v_shape);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), scale, q_shape, k_shape, v_shape);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), scale, q_shape, k_shape, v_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

}
} // namespace llaisys::ops::cpu