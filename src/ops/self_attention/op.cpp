#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
    /** 
     *@brief  Self  attention
     *  
     *@param attn_val: [seqlen, nhead, dv]
     *@param q: [seqlen, nhead, d]
     *@param k: [totallen, nkvhead, d]
     *@param v: [totallen, nkvhead, dv]
     *@param scale: scale
     */
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->ndim() == 3, "Attention: attn_val must be 3-dimensional");
    ASSERT(q->ndim() == 3, "Attention: q must be 3-dimensional");
    ASSERT(k->ndim() == 3, "Attention: k must be 3-dimensional");
    ASSERT(v->ndim() == 3, "Attention: v must be 3-dimensional");
    
    ASSERT(q->shape()[2] == k->shape()[2], "Attention: q and k must have the same dimension");
    ASSERT(k->shape()[0] == v->shape()[0], "Attention: k and v must have the same total length");
    ASSERT(v->shape()[1] == k->shape()[1], "Attention: k and v must have the same number of heads");
    ASSERT(attn_val->shape()[0] == q->shape()[0], "Attention: attn_val and q must have the same length");
    ASSERT(attn_val->shape()[1] == q->shape()[1], "Attention: attn_val and q  must have the same number of heads");
    ASSERT(attn_val->shape()[2] == v->shape()[2], "Attention: attn_val and v must have the same dimension");

    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), scale, q->shape(), k->shape(), v->shape());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
