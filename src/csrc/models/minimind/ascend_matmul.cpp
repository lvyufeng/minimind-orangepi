#include "ascend_matmul.h"

#include <algorithm>
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
#include <vector>

#if defined(MINIMIND_USE_ASCEND)
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_addmm.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_mv.h>
#endif

namespace minimind::model {

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

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = other.data_;
    bytes_ = other.bytes_;
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() { release(); }

void DeviceBuffer::reset(std::size_t bytes) {
  release();
  if (bytes == 0) {
    return;
  }
  check_acl(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc failed");
  bytes_ = bytes;
}

void DeviceBuffer::reserve(std::size_t bytes) {
  if (bytes <= bytes_) {
    return;
  }
  if (data_ != nullptr) {
    check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream before DeviceBuffer grow failed");
  }
  reset(bytes);
}

void DeviceBuffer::release() noexcept {
  if (data_ != nullptr) {
    (void)aclrtFree(data_);
    data_ = nullptr;
    bytes_ = 0;
  }
}

int64_t DeviceTensor::elements() const {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return shape.empty() ? 0 : count;
}

TensorHandle::TensorHandle(const int64_t* dims,
                           const int64_t* strides,
                           uint64_t dims_num,
                           void* data,
                           aclDataType dtype) {
  tensor_ = aclCreateTensor(dims, dims_num, dtype, strides, 0, ACL_FORMAT_ND, dims, dims_num, data);
  if (tensor_ == nullptr) {
    throw std::runtime_error("aclCreateTensor failed");
  }
}

TensorHandle::~TensorHandle() {
  if (tensor_ != nullptr) {
    (void)aclDestroyTensor(tensor_);
  }
}

AscendRuntime::AscendRuntime() {
  const char* custom_opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
#if defined(MINIMIND_CUSTOM_OPP_ROOT)
  if (custom_opp_path == nullptr || custom_opp_path[0] == '\0' || !std::filesystem::exists(custom_opp_path)) {
    std::filesystem::path custom_root = MINIMIND_CUSTOM_OPP_ROOT;
    const auto vendor_root = custom_root / "vendors" / "minimind_orangepi";
    if (std::filesystem::exists(vendor_root)) {
      custom_root = vendor_root;
    }
    setenv("ASCEND_CUSTOM_OPP_PATH", custom_root.c_str(), 1);
  }
#endif
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

AscendRuntime::~AscendRuntime() {
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

aclrtStream AscendRuntime::stream() const noexcept { return stream_; }

AscendRuntime& runtime() {
  static AscendRuntime instance;
  return instance;
}

DeviceTensor make_device_tensor(const std::vector<int64_t>& shape, aclDataType dtype) {
  DeviceTensor tensor;
  tensor.shape = shape;
  tensor.dtype = dtype;
  tensor.strides.resize(shape.size());
  int64_t stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    tensor.strides[i] = stride;
    stride *= shape[i];
  }
  const std::size_t element_size = (dtype == ACL_INT32) ? sizeof(int32_t) : sizeof(uint16_t);
  tensor.buffer.reset(static_cast<std::size_t>(tensor.elements()) * element_size);
  return tensor;
}

DeviceTensor upload_tensor_fp16(const std::vector<float>& values, const std::vector<int64_t>& shape) {
  DeviceTensor tensor = make_device_tensor(shape, ACL_FLOAT16);
  if (static_cast<int64_t>(values.size()) != tensor.elements()) {
    throw std::invalid_argument("upload_tensor_fp16 size mismatch");
  }
  std::vector<uint16_t> half_values(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    half_values[i] = float_to_half(values[i]);
  }
  check_acl(aclrtMemcpy(tensor.data(), tensor.bytes(), half_values.data(), half_values.size() * sizeof(uint16_t),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy upload_tensor_fp16 H2D failed");
  return tensor;
}

DeviceTensor upload_tensor_int32(const std::vector<int32_t>& values, const std::vector<int64_t>& shape) {
  DeviceTensor tensor = make_device_tensor(shape, ACL_INT32);
  if (static_cast<int64_t>(values.size()) != tensor.elements()) {
    throw std::invalid_argument("upload_tensor_int32 size mismatch");
  }
  check_acl(aclrtMemcpy(tensor.data(), tensor.bytes(), values.data(), values.size() * sizeof(int32_t),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy upload_tensor_int32 H2D failed");
  return tensor;
}

void copy_tensor_to_host_fp16(const DeviceTensor& tensor, std::vector<float>& values) {
  const std::size_t count = static_cast<std::size_t>(tensor.elements());
  std::vector<uint16_t> half_values(count);
  check_acl(aclrtMemcpy(half_values.data(), half_values.size() * sizeof(uint16_t), tensor.data(),
                        half_values.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy copy_tensor_to_host_fp16 D2H failed");
  values.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = half_to_float(half_values[i]);
  }
}

void copy_device_tensor(const DeviceTensor& input, DeviceTensor& out) {
  if (input.elements() != out.elements() || input.dtype != out.dtype) {
    throw std::invalid_argument("copy_device_tensor shape mismatch");
  }
  check_acl(aclrtMemcpyAsync(out.data(), out.bytes(), input.data(), input.bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE,
                             runtime().stream()),
            "aclrtMemcpyAsync copy_device_tensor D2D failed");
}

DeviceMatrix upload_matrix_transposed_fp16(const std::vector<float>& matrix, int64_t rows, int64_t cols) {
  if (static_cast<int64_t>(matrix.size()) != rows * cols) {
    throw std::invalid_argument("upload_matrix_transposed_fp16 size mismatch");
  }
  std::vector<uint16_t> matrix_half(static_cast<std::size_t>(rows * cols));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      matrix_half[static_cast<std::size_t>(col * rows + row)] =
          float_to_half(matrix[static_cast<std::size_t>(row * cols + col)]);
    }
  }

  DeviceMatrix uploaded;
  uploaded.rows = rows;
  uploaded.cols = cols;
  uploaded.device.reset(matrix_half.size() * sizeof(uint16_t));
  check_acl(aclrtMemcpy(uploaded.device.data(), uploaded.device.bytes(), matrix_half.data(),
                        matrix_half.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy matrix H2D failed");
  return uploaded;
}

namespace {

struct CachedMatrix {
  DeviceMatrix matrix;
};

struct DeviceMatmulScratch {
  DeviceBuffer workspace;
  DeviceBuffer argmax_workspace;
  DeviceBuffer token_device;
};

struct MatvecScratch {
  DeviceBuffer input_device;
  DeviceBuffer output_device;
  DeviceBuffer workspace;
  DeviceBuffer token_device;
  DeviceBuffer argmax_workspace;
  std::vector<uint16_t> input_half;
  std::vector<uint16_t> output_half;
};

struct MatmulScratch {
  DeviceBuffer input_device;
  DeviceBuffer output_device;
  DeviceBuffer workspace;
  std::vector<uint16_t> input_half;
  std::vector<uint16_t> output_half;
};

DeviceMatmulScratch& device_matmul_scratch() {
  static DeviceMatmulScratch* scratch = new DeviceMatmulScratch();
  return *scratch;
}

MatvecScratch& matvec_scratch() {
  thread_local MatvecScratch scratch;
  return scratch;
}

MatmulScratch& matmul_scratch() {
  thread_local MatmulScratch scratch;
  return scratch;
}

const CachedMatrix& cached_matrix(const std::vector<float>& matrix, int64_t rows, int64_t cols) {
  static std::mutex mutex;
  static std::unordered_map<const float*, CachedMatrix> cache;

  std::lock_guard<std::mutex> lock(mutex);
  const float* key = matrix.data();
  auto it = cache.find(key);
  if (it != cache.end()) {
    if (it->second.matrix.rows != rows || it->second.matrix.cols != cols) {
      throw std::runtime_error("cached matrix shape mismatch");
    }
    return it->second;
  }

  CachedMatrix cached;
  cached.matrix = upload_matrix_transposed_fp16(matrix, rows, cols);
  auto [inserted, _] = cache.emplace(key, std::move(cached));
  return inserted->second;
}

}  // namespace

#endif

bool cube_matvec_available() {
#if defined(MINIMIND_USE_ASCEND)
  return true;
#else
  return false;
#endif
}

#if defined(MINIMIND_USE_ASCEND)
void device_matmul(const DeviceMatrix& matrix, const DeviceTensor& input, int64_t batch, DeviceTensor& out) {
  if (matrix.rows <= 0 || matrix.cols <= 0 || input.elements() != batch * matrix.cols ||
      out.elements() != batch * matrix.rows) {
    throw std::invalid_argument("device_matmul shape mismatch");
  }

  const int64_t input_dims[2] = {batch, matrix.cols};
  const int64_t matrix_dims[2] = {matrix.cols, matrix.rows};
  const int64_t out_dims[2] = {batch, matrix.rows};
  const int64_t input_strides[2] = {matrix.cols, 1};
  const int64_t matrix_strides[2] = {matrix.rows, 1};
  const int64_t out_strides[2] = {matrix.rows, 1};
  TensorHandle lhs(input_dims, input_strides, 2, input.data());
  TensorHandle rhs(matrix_dims, matrix_strides, 2, matrix.data());
  TensorHandle output(out_dims, out_strides, 2, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  constexpr int8_t cube_math_type = 0;
  check_aclnn(aclnnMmGetWorkspaceSize(lhs.get(), rhs.get(), output.get(), cube_math_type, &workspace_size, &executor),
              "aclnnMmGetWorkspaceSize device_matmul failed");
  auto& scratch = device_matmul_scratch();
  scratch.workspace.reserve(workspace_size);
  check_aclnn(aclnnMm(scratch.workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMm device_matmul failed");
}


void device_matmul_add(const DeviceMatrix& matrix,
                       const DeviceTensor& input,
                       const DeviceTensor& residual,
                       int64_t batch,
                       DeviceTensor& out) {
  if (matrix.rows <= 0 || matrix.cols <= 0 || input.elements() != batch * matrix.cols ||
      residual.elements() != batch * matrix.rows || out.elements() != batch * matrix.rows) {
    throw std::invalid_argument("device_matmul_add shape mismatch");
  }

  const int64_t input_dims[2] = {batch, matrix.cols};
  const int64_t matrix_dims[2] = {matrix.cols, matrix.rows};
  const int64_t out_dims[2] = {batch, matrix.rows};
  const int64_t input_strides[2] = {matrix.cols, 1};
  const int64_t matrix_strides[2] = {matrix.rows, 1};
  const int64_t out_strides[2] = {matrix.rows, 1};
  TensorHandle residual_tensor(out_dims, out_strides, 2, residual.data());
  TensorHandle lhs(input_dims, input_strides, 2, input.data());
  TensorHandle rhs(matrix_dims, matrix_strides, 2, matrix.data());
  TensorHandle output(out_dims, out_strides, 2, out.data());

  float beta_value = 1.0F;
  float alpha_value = 1.0F;
  aclScalar* beta = aclCreateScalar(&beta_value, ACL_FLOAT);
  aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
  if (beta == nullptr || alpha == nullptr) {
    if (beta != nullptr) {
      (void)aclDestroyScalar(beta);
    }
    if (alpha != nullptr) {
      (void)aclDestroyScalar(alpha);
    }
    throw std::runtime_error("aclCreateScalar addmm failed");
  }

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  constexpr int8_t cube_math_type = 0;
  try {
    check_aclnn(aclnnAddmmGetWorkspaceSize(residual_tensor.get(), lhs.get(), rhs.get(), beta, alpha, output.get(),
                                           cube_math_type, &workspace_size, &executor),
                "aclnnAddmmGetWorkspaceSize device_matmul_add failed");
    auto& scratch = device_matmul_scratch();
    scratch.workspace.reserve(workspace_size);
    check_aclnn(aclnnAddmm(scratch.workspace.data(), workspace_size, executor, runtime().stream()),
                "aclnnAddmm device_matmul_add failed");
  } catch (...) {
    (void)aclDestroyScalar(beta);
    (void)aclDestroyScalar(alpha);
    throw;
  }
  check_aclnn(aclDestroyScalar(beta), "aclDestroyScalar beta failed");
  check_aclnn(aclDestroyScalar(alpha), "aclDestroyScalar alpha failed");
}
int32_t device_matmul_argmax(const DeviceMatrix& matrix, const DeviceTensor& input, DeviceTensor& logits) {
  if (matrix.rows <= 0 || matrix.cols <= 0 || input.elements() != matrix.cols || logits.elements() != matrix.rows) {
    throw std::invalid_argument("device_matmul_argmax shape mismatch");
  }

  device_matmul(matrix, input, 1, logits);

  auto& scratch = device_matmul_scratch();
  scratch.token_device.reserve(sizeof(int32_t));
  const int64_t token_dims[1] = {1};
  const int64_t token_strides[1] = {1};
  TensorHandle token_tensor(token_dims, token_strides, 1, scratch.token_device.data(), ACL_INT32);

  const int64_t logits_dims[2] = {1, matrix.rows};
  const int64_t logits_strides[2] = {matrix.rows, 1};
  TensorHandle logits_tensor(logits_dims, logits_strides, 2, logits.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnArgMaxGetWorkspaceSize(logits_tensor.get(), 1, false, token_tensor.get(), &workspace_size, &executor),
              "aclnnArgMaxGetWorkspaceSize device_matmul_argmax failed");
  scratch.argmax_workspace.reserve(workspace_size);
  check_aclnn(aclnnArgMax(scratch.argmax_workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnArgMax device_matmul_argmax failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream device_matmul_argmax failed");

  int32_t token = 0;
  check_acl(aclrtMemcpy(&token, sizeof(token), scratch.token_device.data(), sizeof(token), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy token D2H failed");
  return token;
}

int32_t device_matmul_argmax(const DeviceMatrix& matrix, const DeviceTensor& input) {
  DeviceTensor logits = make_device_tensor({1, matrix.rows});
  return device_matmul_argmax(matrix, input, logits);
}
#endif

bool cube_matmul_available() {
#if defined(MINIMIND_USE_ASCEND)
  return true;
#else
  return false;
#endif
}

std::vector<float> cube_matmul(const std::vector<float>& matrix,
                               int64_t rows,
                               int64_t cols,
                               const std::vector<float>& input,
                               int64_t batch) {
  if (static_cast<int64_t>(matrix.size()) != rows * cols || static_cast<int64_t>(input.size()) != batch * cols) {
    throw std::invalid_argument("invalid cube_matmul shape");
  }

#if defined(MINIMIND_USE_ASCEND)
  AscendRuntime& rt = runtime();
  const CachedMatrix& weight = cached_matrix(matrix, rows, cols);

  auto& scratch = matmul_scratch();
  scratch.input_half.resize(static_cast<std::size_t>(batch * cols));
  for (int64_t i = 0; i < batch * cols; ++i) {
    scratch.input_half[static_cast<std::size_t>(i)] = float_to_half(input[static_cast<std::size_t>(i)]);
  }
  scratch.output_half.resize(static_cast<std::size_t>(batch * rows));

  scratch.input_device.reserve(scratch.input_half.size() * sizeof(uint16_t));
  scratch.output_device.reserve(scratch.output_half.size() * sizeof(uint16_t));
  check_acl(aclrtMemcpy(scratch.input_device.data(), scratch.input_half.size() * sizeof(uint16_t),
                        scratch.input_half.data(), scratch.input_half.size() * sizeof(uint16_t),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy matmul input H2D failed");

  const int64_t input_dims[2] = {batch, cols};
  const int64_t matrix_dims[2] = {cols, rows};
  const int64_t out_dims[2] = {batch, rows};
  const int64_t input_strides[2] = {cols, 1};
  const int64_t matrix_strides[2] = {rows, 1};
  const int64_t out_strides[2] = {rows, 1};
  TensorHandle lhs(input_dims, input_strides, 2, scratch.input_device.data());
  TensorHandle rhs(matrix_dims, matrix_strides, 2, weight.matrix.data());
  TensorHandle out(out_dims, out_strides, 2, scratch.output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  constexpr int8_t cube_math_type = 0;
  check_aclnn(aclnnMmGetWorkspaceSize(lhs.get(), rhs.get(), out.get(), cube_math_type, &workspace_size, &executor),
              "aclnnMmGetWorkspaceSize matmul failed");
  scratch.workspace.reserve(workspace_size);
  check_aclnn(aclnnMm(scratch.workspace.data(), workspace_size, executor, rt.stream()), "aclnnMm matmul failed");
  check_acl(aclrtSynchronizeStream(rt.stream()), "aclrtSynchronizeStream matmul failed");
  check_acl(aclrtMemcpy(scratch.output_half.data(), scratch.output_half.size() * sizeof(uint16_t),
                        scratch.output_device.data(), scratch.output_half.size() * sizeof(uint16_t),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy matmul output D2H failed");

  std::vector<float> output(static_cast<std::size_t>(batch * rows));
  for (int64_t i = 0; i < batch * rows; ++i) {
    output[static_cast<std::size_t>(i)] = half_to_float(scratch.output_half[static_cast<std::size_t>(i)]);
  }
  return output;
#else
  throw std::runtime_error("cube_matmul requires MINIMIND_USE_ASCEND");
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

  auto& scratch = matvec_scratch();
  scratch.input_half.resize(static_cast<std::size_t>(cols));
  for (int64_t i = 0; i < cols; ++i) {
    scratch.input_half[static_cast<std::size_t>(i)] = float_to_half(input[static_cast<std::size_t>(i)]);
  }
  scratch.output_half.resize(static_cast<std::size_t>(rows));

  scratch.input_device.reserve(scratch.input_half.size() * sizeof(uint16_t));
  scratch.output_device.reserve(scratch.output_half.size() * sizeof(uint16_t));
  check_acl(aclrtMemcpy(scratch.input_device.data(), scratch.input_half.size() * sizeof(uint16_t), scratch.input_half.data(),
                        scratch.input_half.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy input H2D failed");

  const int64_t input_dims[2] = {1, cols};
  const int64_t matrix_dims[2] = {cols, rows};
  const int64_t out_dims[2] = {1, rows};
  const int64_t input_strides[2] = {cols, 1};
  const int64_t matrix_strides[2] = {rows, 1};
  const int64_t out_strides[2] = {rows, 1};
  TensorHandle lhs(input_dims, input_strides, 2, scratch.input_device.data());
  TensorHandle rhs(matrix_dims, matrix_strides, 2, weight.matrix.data());
  TensorHandle out(out_dims, out_strides, 2, scratch.output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  constexpr int8_t cube_math_type = 0;
  check_aclnn(aclnnMmGetWorkspaceSize(lhs.get(), rhs.get(), out.get(), cube_math_type, &workspace_size, &executor),
              "aclnnMmGetWorkspaceSize failed");
  scratch.workspace.reserve(workspace_size);
  check_aclnn(aclnnMm(scratch.workspace.data(), workspace_size, executor, rt.stream()), "aclnnMm failed");
  check_acl(aclrtSynchronizeStream(rt.stream()), "aclrtSynchronizeStream failed");
  check_acl(aclrtMemcpy(scratch.output_half.data(), scratch.output_half.size() * sizeof(uint16_t),
                        scratch.output_device.data(), scratch.output_half.size() * sizeof(uint16_t),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy output D2H failed");

  std::vector<float> output(static_cast<std::size_t>(rows));
  for (int64_t i = 0; i < rows; ++i) {
    output[static_cast<std::size_t>(i)] = half_to_float(scratch.output_half[static_cast<std::size_t>(i)]);
  }
  return output;
#else
  return cpu_matvec(matrix, rows, cols, input);
#endif
}

int32_t cube_matvec_argmax(const std::vector<float>& matrix,
                           int64_t rows,
                           int64_t cols,
                           const std::vector<float>& input) {
  if (static_cast<int64_t>(matrix.size()) != rows * cols || static_cast<int64_t>(input.size()) != cols) {
    throw std::invalid_argument("invalid cube_matvec_argmax shape");
  }

#if defined(MINIMIND_USE_ASCEND)
  AscendRuntime& rt = runtime();
  const CachedMatrix& weight = cached_matrix(matrix, rows, cols);

  auto& scratch = matvec_scratch();
  scratch.input_half.resize(static_cast<std::size_t>(cols));
  for (int64_t i = 0; i < cols; ++i) {
    scratch.input_half[static_cast<std::size_t>(i)] = float_to_half(input[static_cast<std::size_t>(i)]);
  }

  scratch.input_device.reserve(scratch.input_half.size() * sizeof(uint16_t));
  scratch.output_device.reserve(static_cast<std::size_t>(rows) * sizeof(uint16_t));
  scratch.token_device.reserve(sizeof(int32_t));
  check_acl(aclrtMemcpy(scratch.input_device.data(), scratch.input_half.size() * sizeof(uint16_t), scratch.input_half.data(),
                        scratch.input_half.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy input H2D failed");

  const int64_t input_dims[2] = {1, cols};
  const int64_t matrix_dims[2] = {cols, rows};
  const int64_t out_dims[2] = {1, rows};
  const int64_t input_strides[2] = {cols, 1};
  const int64_t matrix_strides[2] = {rows, 1};
  const int64_t out_strides[2] = {rows, 1};
  TensorHandle lhs(input_dims, input_strides, 2, scratch.input_device.data());
  TensorHandle rhs(matrix_dims, matrix_strides, 2, weight.matrix.data());
  TensorHandle out(out_dims, out_strides, 2, scratch.output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  constexpr int8_t cube_math_type = 0;
  check_aclnn(aclnnMmGetWorkspaceSize(lhs.get(), rhs.get(), out.get(), cube_math_type, &workspace_size, &executor),
              "aclnnMmGetWorkspaceSize failed");
  scratch.workspace.reserve(workspace_size);
  check_aclnn(aclnnMm(scratch.workspace.data(), workspace_size, executor, rt.stream()), "aclnnMm failed");

  const int64_t token_dims[1] = {1};
  const int64_t token_strides[1] = {1};
  TensorHandle token_tensor(token_dims, token_strides, 1, scratch.token_device.data(), ACL_INT32);
  workspace_size = 0;
  executor = nullptr;
  check_aclnn(aclnnArgMaxGetWorkspaceSize(out.get(), 1, false, token_tensor.get(), &workspace_size, &executor),
              "aclnnArgMaxGetWorkspaceSize failed");
  scratch.argmax_workspace.reserve(workspace_size);
  check_aclnn(aclnnArgMax(scratch.argmax_workspace.data(), workspace_size, executor, rt.stream()), "aclnnArgMax failed");
  check_acl(aclrtSynchronizeStream(rt.stream()), "aclrtSynchronizeStream failed");

  int32_t token = 0;
  check_acl(aclrtMemcpy(&token, sizeof(token), scratch.token_device.data(), sizeof(token), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy token D2H failed");
  return token;
#else
  const auto logits = cpu_matvec(matrix, rows, cols, input);
  return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
#endif
}

}  // namespace minimind::model
