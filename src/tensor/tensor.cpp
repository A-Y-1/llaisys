#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

void Tensor::shapeDebug() const {
    std::cout<<"Tensor shape: [";
    for(size_t i=0;i<this->ndim();i++){
        std::cout <<this->shape()[i]<<" ";
    }
    std::cout<<"]"<<std::endl<<" stride:[";
    for(size_t i=0;i<this->ndim();i++){
        std::cout <<this->strides()[i]<<" ";
    }
    std::cout<<"]"<<std::endl;
}

bool Tensor::isContiguous() const {
    auto &shape_ = shape();
    auto &strides_ = strides();
    ptrdiff_t expected_stride = 1;
    for(int64_t i = (int64_t)ndim() - 1; i >= 0; i--){
        if(strides_[i] != expected_stride){
            return false;
        }
        expected_stride *= shape_[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    ASSERT(order.size() == shape().size(), "Tensor::permute: The number of dimensions in the order vector unequal to tensor dimensions");
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;

    auto &old_shape = shape();
    auto &old_strides = strides();
    new_meta.shape.resize(old_shape.size());
    new_meta.strides.resize(old_strides.size());
    for(size_t i=0;i<old_shape.size();i++){
        ASSERT(order[i] < old_shape.size(), "Tensor::permute: Dimension index out of range");
        new_meta.shape[i] = old_shape[order[i]];
        new_meta.strides[i] = old_strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

std::string vec_to_string(const std::vector<size_t>& vec) {
    std::string s = "(";
    for(size_t i=0; i<vec.size(); ++i) s += std::to_string(vec[i]) + (i==vec.size()-1 ? "" : ", ");
    return s + ")";
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t numel_new = 1;
    for (auto s: shape){
        numel_new *= s;
    }
    if(numel_new != numel()){
        std::string msg = "[ERROR] Tensor::view: element count mismatch. Original: " + 
                            std::to_string(numel()) + ", Target: " + std::to_string(numel_new);
        throw std::runtime_error(msg);
    }
    if(!isContiguous()){
        std::string msg = "[ERROR] Tensor::view: tensor must be contiguous."; 
        throw std::runtime_error(msg);
    }

    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= shape[i];
    }

    TensorMeta new_meta{this->dtype(), shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    ASSERT(dim<ndim(), "Tensor::slice: dim exceed tensor dimensions");
    ASSERT(start >= 0 && end <= shape()[dim] && end >= start, "Tensor:slice: invalid slice range");
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;
    size_t new_offset = _offset + start * strides()[dim] * elementSize();
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT((src_ != nullptr), " source pointer is null");
    ASSERT(isContiguous(), "Tensor::load: tensor is not contigous");
    
    size_t bytes = numel() * elementSize();
    ASSERT(bytes + _offset <= _storage->size(), "Tensor::load: load size exceeds storage");

    std::byte *dst = data();
    if(_storage->isHost()){
        std::memcpy(dst, src_, bytes);
    }else{
        core::context().runtime().api()->memcpy_sync(
            dst,
            src_, 
            bytes,
            LLAISYS_MEMCPY_H2D
        );
    }
    return;
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
