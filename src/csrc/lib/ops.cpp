#include "minimind_orangepi/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace minimind {

namespace {

bool same_shape(const TensorShape& lhs, const TensorShape& rhs) {
  return lhs.dims() == rhs.dims();
}

void require_float32(const HostTensor& tensor, const char* name) {
  if (tensor.dtype() != DType::kFloat32) {
    throw std::invalid_argument(std::string(name) + " must be float32");
  }
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

HostTensor rms_norm(const HostTensor& input, const HostTensor& weight, float eps) {
  require_float32(input, "rms_norm input");
  require_float32(weight, "rms_norm weight");
  if (weight.shape().rank() != 1) {
    throw std::invalid_argument("rms_norm weight must be rank 1");
  }
  const int64_t hidden_size = weight.shape().dim(0);
  if (hidden_size <= 0 || input.shape().numel() % hidden_size != 0) {
    throw std::invalid_argument("rms_norm input size must be divisible by hidden size");
  }

  HostTensor output(input.shape(), DType::kFloat32);
  const auto* x = input.data_as<float>();
  const auto* w = weight.data_as<float>();
  auto* y = output.data_as<float>();
  const int64_t rows = input.shape().numel() / hidden_size;
  for (int64_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (int64_t col = 0; col < hidden_size; ++col) {
      const float value = x[row * hidden_size + col];
      sum += value * value;
    }
    const float scale = 1.0F / std::sqrt(sum / static_cast<float>(hidden_size) + eps);
    for (int64_t col = 0; col < hidden_size; ++col) {
      y[row * hidden_size + col] = x[row * hidden_size + col] * scale * w[col];
    }
  }
  return output;
}

HostTensor swiglu(const HostTensor& gate, const HostTensor& up) {
  require_float32(gate, "swiglu gate");
  require_float32(up, "swiglu up");
  if (!same_shape(gate.shape(), up.shape())) {
    throw std::invalid_argument("swiglu requires matching shapes");
  }

  HostTensor output(gate.shape(), DType::kFloat32);
  const auto* g = gate.data_as<float>();
  const auto* u = up.data_as<float>();
  auto* y = output.data_as<float>();
  for (std::size_t i = 0; i < gate.numel(); ++i) {
    y[i] = (g[i] / (1.0F + std::exp(-g[i]))) * u[i];
  }
  return output;
}

}  // namespace minimind
