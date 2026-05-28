#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimind {

enum class DType {
  kFloat32,
  kFloat16,
  kInt32,
  kUInt8,
};

std::string dtype_name(DType dtype);
std::size_t dtype_size(DType dtype);

class TensorShape {
 public:
  TensorShape() = default;
  explicit TensorShape(std::vector<int64_t> dims);

  const std::vector<int64_t>& dims() const noexcept { return dims_; }
  std::size_t rank() const noexcept { return dims_.size(); }
  int64_t dim(std::size_t index) const;
  int64_t numel() const noexcept;
  std::string str() const;

 private:
  std::vector<int64_t> dims_;
};

class HostTensor {
 public:
  HostTensor() = default;
  HostTensor(TensorShape shape, DType dtype);

  const TensorShape& shape() const noexcept { return shape_; }
  DType dtype() const noexcept { return dtype_; }
  std::size_t bytes() const noexcept { return data_.size(); }
  std::size_t numel() const noexcept { return static_cast<std::size_t>(shape_.numel()); }

  void* data() noexcept { return data_.data(); }
  const void* data() const noexcept { return data_.data(); }

  template <typename T>
  T* data_as() noexcept {
    return reinterpret_cast<T*>(data_.data());
  }

  template <typename T>
  const T* data_as() const noexcept {
    return reinterpret_cast<const T*>(data_.data());
  }

 private:
  TensorShape shape_;
  DType dtype_ = DType::kFloat32;
  std::vector<std::uint8_t> data_;
};

}  // namespace minimind
