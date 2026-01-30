#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensor must be contigous.");
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "Linear: tensor dim must be 2.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: input and weight tensor's dimension mismatch.");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "Linear: output dimension mismatch with input and weight.");

    std::byte *bias_ptr = nullptr;
    if(bias){
        CHECK_SAME_DEVICE(in, bias);
        CHECK_SAME_DTYPE(in->dtype(), out->dtype());
        ASSERT(bias->ndim() == 1 || bias->ndim() == 0, "Linear: tensor dim must be 0 or 1.");
        ASSERT(bias->isContiguous(), "Linear: all tensor must be contigous.");
        bias_ptr = bias->data();
    }
    size_t bias_dim = bias_ptr == nullptr ? 0 : bias->ndim();

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, in->shape()[0], in->shape()[1], weight->shape()[0], out->dtype(), bias_dim);
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
