#include "minimind_orangepi/tensor.h"

#include <sstream>

namespace minimind {

std::string dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kInt32:
      return "int32";
    case DType::kUInt8:
      return "uint8";
  }
  throw std::invalid_argument("unknown dtype");
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
      return 2;
    case DType::kInt32:
      return 4;
    case DType::kUInt8:
      return 1;
  }
  throw std::invalid_argument("unknown dtype");
}

TensorShape::TensorShape(std::vector<int64_t> dims) : dims_(std::move(dims)) {
  for (int64_t dim : dims_) {
    if (dim < 0) {
      throw std::invalid_argument("tensor dimensions must be non-negative");
    }
  }
}

int64_t TensorShape::dim(std::size_t index) const {
  if (index >= dims_.size()) {
    throw std::out_of_range("tensor dimension index out of range");
  }
  return dims_[index];
}

int64_t TensorShape::numel() const noexcept {
  if (dims_.empty()) {
    return 1;
  }
  return std::accumulate(dims_.begin(), dims_.end(), int64_t{1},
                         [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
}

std::string TensorShape::str() const {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << dims_[i];
  }
  out << ']';
  return out.str();
}

HostTensor::HostTensor(TensorShape shape, DType dtype)
    : shape_(std::move(shape)), dtype_(dtype) {
  const int64_t elements = shape_.numel();
  if (elements < 0) {
    throw std::invalid_argument("tensor element count overflowed");
  }
  data_.resize(static_cast<std::size_t>(elements) * dtype_size(dtype_));
}

}  // namespace minimind
