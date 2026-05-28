#include "ascend_matmul.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#if defined(MINIMIND_USE_ASCEND)
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_mv.h>
#endif

namespace minimind::model {
namespace {

uint16_t float_to_half(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half_mantissa = mantissa >> shift;
    if (((mantissa >> (shift - 1)) & 1U) != 0U) {
      half_mantissa += 1U;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }

  mantissa += 0x1000U;
  if ((mantissa & 0x800000U) != 0U) {
    mantissa = 0;
    exponent += 1;
    if (exponent >= 31) {
      return static_cast<uint16_t>(sign | 0x7C00U);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa >> 13U));
}

float half_to_float(uint16_t value) {
  const uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16U;
  uint32_t exponent = (value >> 10U) & 0x1FU;
  uint32_t mantissa = value & 0x03FFU;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        exponent -= 1U;
      }
      mantissa &= 0x03FFU;
      bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
  }

  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

#if defined(MINIMIND_USE_ASCEND)

void check_acl(aclError status, const char* message) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(message) + ": " + std::to_string(static_cast<int>(status)));
  }
}

void check_aclnn(aclnnStatus status, const char* message) {
  if (status != OK) {
    throw std::runtime_error(std::string(message) + ": " + std::to_string(status));
  }
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) { reset(bytes); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      bytes_ = other.bytes_;
      other.data_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  void* data() const noexcept { return data_; }
  std::size_t bytes() const noexcept { return bytes_; }

  void reset(std::size_t bytes) {
    release();
    if (bytes == 0) {
      return;
    }
    check_acl(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed");
    bytes_ = bytes;
  }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      (void)aclrtFree(data_);
      data_ = nullptr;
      bytes_ = 0;
    }
  }

  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

class TensorHandle {
 public:
  TensorHandle(const int64_t* dims, const int64_t* strides, uint64_t dims_num, void* data) {
    tensor_ = aclCreateTensor(dims, dims_num, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, dims, dims_num, data);
    if (tensor_ == nullptr) {
      throw std::runtime_error("aclCreateTensor failed");
    }
  }
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;
  ~TensorHandle() {
    if (tensor_ != nullptr) {
      (void)aclDestroyTensor(tensor_);
    }
  }
  aclTensor* get() const noexcept { return tensor_; }

 private:
  aclTensor* tensor_ = nullptr;
};

class AscendRuntime {
 public:
  AscendRuntime() {
    const char* opp_path = std::getenv("ASCEND_OPP_PATH");
    if (opp_path == nullptr || opp_path[0] == '\0' || !std::filesystem::exists(opp_path)) {
      setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/cann/opp", 1);
    }
    check_acl(aclInit(nullptr), "aclInit failed");
    initialized_acl_ = true;
    check_aclnn(aclnnInit(nullptr), "aclnnInit failed");
    initialized_aclnn_ = true;
    check_acl(aclrtSetDevice(0), "aclrtSetDevice failed");
    device_set_ = true;
    check_acl(aclrtCreateStream(&stream_), "aclrtCreateStream failed");
  }

  AscendRuntime(const AscendRuntime&) = delete;
  AscendRuntime& operator=(const AscendRuntime&) = delete;

  ~AscendRuntime() {
    if (stream_ != nullptr) {
      (void)aclrtDestroyStream(stream_);
    }
    if (device_set_) {
      (void)aclrtResetDevice(0);
    }
    if (initialized_aclnn_) {
      (void)aclnnFinalize();
    }
    if (initialized_acl_) {
      (void)aclFinalize();
    }
  }

  aclrtStream stream() const noexcept { return stream_; }

 private:
  bool initialized_acl_ = false;
  bool initialized_aclnn_ = false;
  bool device_set_ = false;
  aclrtStream stream_ = nullptr;
};

struct CachedMatrix {
  int64_t rows = 0;
  int64_t cols = 0;
  DeviceBuffer device;
};

AscendRuntime& runtime() {
  static AscendRuntime instance;
  return instance;
}

const CachedMatrix& cached_matrix(const std::vector<float>& matrix, int64_t rows, int64_t cols) {
  static std::mutex mutex;
  static std::unordered_map<const float*, CachedMatrix> cache;

  std::lock_guard<std::mutex> lock(mutex);
  const float* key = matrix.data();
  auto it = cache.find(key);
  if (it != cache.end()) {
    if (it->second.rows != rows || it->second.cols != cols) {
      throw std::runtime_error("cached matrix shape mismatch");
    }
    return it->second;
  }

  std::vector<uint16_t> matrix_half(static_cast<std::size_t>(rows * cols));
  for (int64_t i = 0; i < rows * cols; ++i) {
    matrix_half[static_cast<std::size_t>(i)] = float_to_half(matrix[static_cast<std::size_t>(i)]);
  }

  CachedMatrix cached;
  cached.rows = rows;
  cached.cols = cols;
  cached.device.reset(matrix_half.size() * sizeof(uint16_t));
  check_acl(aclrtMemcpy(cached.device.data(), cached.device.bytes(), matrix_half.data(),
                        matrix_half.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy matrix H2D failed");
  auto [inserted, _] = cache.emplace(key, std::move(cached));
  return inserted->second;
}

#endif

}  // namespace

bool cube_matvec_available() {
#if defined(MINIMIND_USE_ASCEND)
  return true;
#else
  return false;
#endif
}

std::vector<float> cube_matvec(const std::vector<float>& matrix,
                               int64_t rows,
                               int64_t cols,
                               const std::vector<float>& input) {
  if (static_cast<int64_t>(matrix.size()) != rows * cols || static_cast<int64_t>(input.size()) != cols) {
    throw std::invalid_argument("invalid cube_matvec shape");
  }

#if defined(MINIMIND_USE_ASCEND)
  AscendRuntime& rt = runtime();
  const CachedMatrix& weight = cached_matrix(matrix, rows, cols);

  std::vector<uint16_t> input_half(static_cast<std::size_t>(cols));
  for (int64_t i = 0; i < cols; ++i) {
    input_half[static_cast<std::size_t>(i)] = float_to_half(input[static_cast<std::size_t>(i)]);
  }
  std::vector<uint16_t> output_half(static_cast<std::size_t>(rows));

  DeviceBuffer input_device(input_half.size() * sizeof(uint16_t));
  DeviceBuffer output_device(output_half.size() * sizeof(uint16_t));
  check_acl(aclrtMemcpy(input_device.data(), input_device.bytes(), input_half.data(), input_device.bytes(),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy input H2D failed");

  const int64_t matrix_dims[2] = {rows, cols};
  const int64_t input_dims[1] = {cols};
  const int64_t out_dims[1] = {rows};
  const int64_t matrix_strides[2] = {cols, 1};
  const int64_t input_strides[1] = {1};
  const int64_t out_strides[1] = {1};
  TensorHandle lhs(matrix_dims, matrix_strides, 2, weight.device.data());
  TensorHandle rhs(input_dims, input_strides, 1, input_device.data());
  TensorHandle out(out_dims, out_strides, 1, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  constexpr int8_t cube_math_type = 0;
  check_aclnn(aclnnMvGetWorkspaceSize(lhs.get(), rhs.get(), out.get(), cube_math_type, &workspace_size, &executor),
              "aclnnMvGetWorkspaceSize failed");
  DeviceBuffer workspace(workspace_size);
  check_aclnn(aclnnMv(workspace.data(), workspace_size, executor, rt.stream()), "aclnnMv failed");
  check_acl(aclrtSynchronizeStream(rt.stream()), "aclrtSynchronizeStream failed");
  check_acl(aclrtMemcpy(output_half.data(), output_half.size() * sizeof(uint16_t), output_device.data(),
                        output_device.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy output D2H failed");

  std::vector<float> output(static_cast<std::size_t>(rows));
  for (int64_t i = 0; i < rows; ++i) {
    output[static_cast<std::size_t>(i)] = half_to_float(output_half[static_cast<std::size_t>(i)]);
  }
  return output;
#else
  return cpu_matvec(matrix, rows, cols, input);
#endif
}

}  // namespace minimind::model
