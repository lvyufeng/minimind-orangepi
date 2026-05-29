#pragma once

#include <cstdint>
#include <vector>

#if defined(MINIMIND_USE_ASCEND)
#include <acl/acl_rt.h>
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

AscendRuntime& runtime();
#endif

std::vector<float> cube_matvec(const std::vector<float>& matrix,
                               int64_t rows,
                               int64_t cols,
                               const std::vector<float>& input);

int32_t cube_matvec_argmax(const std::vector<float>& matrix,
                           int64_t rows,
                           int64_t cols,
                           const std::vector<float>& input);

bool cube_matvec_available();

}  // namespace minimind::model
