#include "ascend_custom_ops.h"
#include "ascend_matmul.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_mini_mind_rms_norm.h>
#include <aclnn_mini_mind_swi_glu.h>
#endif

namespace minimind::model {

namespace {

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)

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

DeviceBuffer copy_half_to_device(const std::vector<float>& values) {
  std::vector<uint16_t> half_values(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    half_values[i] = float_to_half(values[i]);
  }

  DeviceBuffer device(half_values.size() * sizeof(uint16_t));
  check_acl(aclrtMemcpy(device.data(), device.bytes(), half_values.data(), device.bytes(), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy H2D failed");
  return device;
}

std::vector<float> copy_half_to_host(const DeviceBuffer& device, std::size_t values) {
  std::vector<uint16_t> half_values(values);
  check_acl(aclrtMemcpy(half_values.data(), half_values.size() * sizeof(uint16_t), device.data(), device.bytes(),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy D2H failed");

  std::vector<float> output(values);
  for (std::size_t i = 0; i < values; ++i) {
    output[i] = half_to_float(half_values[i]);
  }
  return output;
}

#endif

}  // namespace

bool custom_ops_available() {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  return cube_matvec_available();
#else
  return false;
#endif
}

std::vector<float> custom_rms_norm(const std::vector<float>& input,
                                   const std::vector<float>& weight,
                                   float eps) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (input.size() != weight.size()) {
    throw std::invalid_argument("custom_rms_norm shape mismatch");
  }
  (void)runtime();

  DeviceBuffer input_device = copy_half_to_device(input);
  DeviceBuffer weight_device = copy_half_to_device(weight);
  DeviceBuffer output_device(input.size() * sizeof(uint16_t));

  const int64_t dims[1] = {static_cast<int64_t>(input.size())};
  const int64_t strides[1] = {1};
  TensorHandle x(dims, strides, 1, input_device.data());
  TensorHandle gamma(dims, strides, 1, weight_device.data());
  TensorHandle out(dims, strides, 1, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindRmsNormGetWorkspaceSize(x.get(), gamma.get(), static_cast<double>(eps), out.get(),
                                                   &workspace_size, &executor),
              "aclnnMiniMindRmsNormGetWorkspaceSize failed");
  DeviceBuffer workspace(workspace_size);
  check_aclnn(aclnnMiniMindRmsNorm(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindRmsNorm failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, input.size());
#else
  (void)eps;
  throw std::runtime_error("custom_rms_norm is unavailable");
#endif
}

std::vector<float> custom_swiglu(const std::vector<float>& gate,
                                 const std::vector<float>& up) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (gate.size() != up.size()) {
    throw std::invalid_argument("custom_swiglu shape mismatch");
  }
  (void)runtime();

  DeviceBuffer gate_device = copy_half_to_device(gate);
  DeviceBuffer up_device = copy_half_to_device(up);
  DeviceBuffer output_device(gate.size() * sizeof(uint16_t));

  const int64_t dims[1] = {static_cast<int64_t>(gate.size())};
  const int64_t strides[1] = {1};
  TensorHandle gate_tensor(dims, strides, 1, gate_device.data());
  TensorHandle up_tensor(dims, strides, 1, up_device.data());
  TensorHandle out(dims, strides, 1, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindSwiGluGetWorkspaceSize(gate_tensor.get(), up_tensor.get(), out.get(),
                                                  &workspace_size, &executor),
              "aclnnMiniMindSwiGluGetWorkspaceSize failed");
  DeviceBuffer workspace(workspace_size);
  check_aclnn(aclnnMiniMindSwiGlu(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindSwiGlu failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, gate.size());
#else
  throw std::runtime_error("custom_swiglu is unavailable");
#endif
}

}  // namespace minimind::model
