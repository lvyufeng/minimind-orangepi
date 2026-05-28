#include "minimind_orangepi/ops.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace minimind {

namespace {

bool same_shape(const TensorShape& lhs, const TensorShape& rhs) {
  return lhs.dims() == rhs.dims();
}

}  // namespace

HostTensor zeros(TensorShape shape, DType dtype) {
  return HostTensor(std::move(shape), dtype);
}

HostTensor add(const HostTensor& lhs, const HostTensor& rhs) {
  if (!same_shape(lhs.shape(), rhs.shape())) {
    throw std::invalid_argument("add requires matching shapes");
  }
  if (lhs.dtype() != rhs.dtype()) {
    throw std::invalid_argument("add requires matching dtypes");
  }

  HostTensor output(lhs.shape(), lhs.dtype());
  switch (lhs.dtype()) {
    case DType::kFloat32: {
      const auto* a = lhs.data_as<float>();
      const auto* b = rhs.data_as<float>();
      auto* out = output.data_as<float>();
      for (std::size_t i = 0; i < lhs.numel(); ++i) {
        out[i] = a[i] + b[i];
      }
      return output;
    }
    case DType::kInt32: {
      const auto* a = lhs.data_as<int32_t>();
      const auto* b = rhs.data_as<int32_t>();
      auto* out = output.data_as<int32_t>();
      for (std::size_t i = 0; i < lhs.numel(); ++i) {
        out[i] = a[i] + b[i];
      }
      return output;
    }
    case DType::kFloat16:
    case DType::kUInt8:
      throw std::invalid_argument("add dtype is not implemented for " + dtype_name(lhs.dtype()));
  }
  throw std::invalid_argument("unknown dtype");
}

}  // namespace minimind
