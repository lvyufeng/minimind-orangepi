#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(MINIMIND_USE_ASCEND)
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#endif

namespace minimind::model {

uint16_t float_to_half(float value);
float half_to_float(uint16_t value);

#if defined(MINIMIND_USE_ASCEND)
class AscendRuntime {
 public:
  AscendRuntime();
  AscendRuntime(const AscendRuntime&) = delete;
  AscendRuntime& operator=(const AscendRuntime&) = delete;
  ~AscendRuntime();

  aclrtStream stream() const noexcept;

 private:
  bool initialized_acl_ = false;
  bool initialized_aclnn_ = false;
  bool device_set_ = false;
  aclrtStream stream_ = nullptr;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) { reset(bytes); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer();

  void* data() const noexcept { return data_; }
  std::size_t bytes() const noexcept { return bytes_; }
  void reset(std::size_t bytes);
  void reserve(std::size_t bytes);

 private:
  void release() noexcept;

  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

struct DeviceTensor {
  DeviceBuffer buffer;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  aclDataType dtype = ACL_FLOAT16;

  void* data() const noexcept { return buffer.data(); }
  std::size_t bytes() const noexcept { return buffer.bytes(); }
  int64_t elements() const;
};

struct DeviceMatrix {
  int64_t rows = 0;
  int64_t cols = 0;
  DeviceBuffer device;

  void* data() const noexcept { return device.data(); }
  std::size_t bytes() const noexcept { return device.bytes(); }
};

class TensorHandle {
 public:
  TensorHandle(const int64_t* dims,
               const int64_t* strides,
               uint64_t dims_num,
               void* data,
               aclDataType dtype = ACL_FLOAT16);
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;
  ~TensorHandle();
  aclTensor* get() const noexcept { return tensor_; }

 private:
  aclTensor* tensor_ = nullptr;
};

void check_acl(aclError status, const char* message);
void check_aclnn(aclnnStatus status, const char* message);
AscendRuntime& runtime();

DeviceTensor make_device_tensor(const std::vector<int64_t>& shape, aclDataType dtype = ACL_FLOAT16);
DeviceTensor upload_tensor_fp16(const std::vector<float>& values, const std::vector<int64_t>& shape);
DeviceTensor upload_tensor_int32(const std::vector<int32_t>& values, const std::vector<int64_t>& shape);
void copy_tensor_to_host_fp16(const DeviceTensor& tensor, std::vector<float>& values);
void copy_device_tensor(const DeviceTensor& input, DeviceTensor& out);
DeviceMatrix upload_matrix_transposed_fp16(const std::vector<float>& matrix, int64_t rows, int64_t cols);
void device_matmul(const DeviceMatrix& matrix, const DeviceTensor& input, int64_t batch, DeviceTensor& out);
void device_matmul_add(const DeviceMatrix& matrix,
                       const DeviceTensor& input,
                       const DeviceTensor& residual,
                       int64_t batch,
                       DeviceTensor& out);
int32_t device_matmul_argmax(const DeviceMatrix& matrix, const DeviceTensor& input);
int32_t device_matmul_argmax(const DeviceMatrix& matrix, const DeviceTensor& input, DeviceTensor& logits);

#else
struct DeviceTensor;
struct DeviceMatrix;
#endif

std::vector<float> cube_matvec(const std::vector<float>& matrix,
                               int64_t rows,
                               int64_t cols,
                               const std::vector<float>& input);

std::vector<float> cube_matmul(const std::vector<float>& matrix,
                               int64_t rows,
                               int64_t cols,
                               const std::vector<float>& input,
                               int64_t batch);

int32_t cube_matvec_argmax(const std::vector<float>& matrix,
                           int64_t rows,
                           int64_t cols,
                           const std::vector<float>& input);

bool cube_matvec_available();
bool cube_matmul_available();

}  // namespace minimind::model
