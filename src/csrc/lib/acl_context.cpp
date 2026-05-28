#include "minimind_orangepi/acl_context.h"

#include <stdexcept>
#include <utility>

#if defined(MINIMIND_USE_ASCEND)
#include <acl/acl.h>
#endif

namespace minimind {

AclContext::AclContext(AclContext&& other) noexcept {
  initialized_ = std::exchange(other.initialized_, false);
  device_id_ = std::exchange(other.device_id_, -1);
  backend_ = std::move(other.backend_);
}

AclContext& AclContext::operator=(AclContext&& other) noexcept {
  if (this != &other) {
    reset();
    initialized_ = std::exchange(other.initialized_, false);
    device_id_ = std::exchange(other.device_id_, -1);
    backend_ = std::move(other.backend_);
  }
  return *this;
}

AclContext::~AclContext() {
  reset();
}

void AclContext::initialize(int32_t device_id) {
  if (initialized_) {
    return;
  }

#if defined(MINIMIND_USE_ASCEND)
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    throw std::runtime_error("aclInit failed");
  }
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    aclFinalize();
    throw std::runtime_error("aclrtSetDevice failed");
  }
  backend_ = "ascend";
#else
  backend_ = "host";
#endif

  device_id_ = device_id;
  initialized_ = true;
}

void AclContext::reset() {
  if (!initialized_) {
    return;
  }

#if defined(MINIMIND_USE_ASCEND)
  aclrtResetDevice(device_id_);
  aclFinalize();
#endif

  initialized_ = false;
  device_id_ = -1;
  backend_ = "host";
}

}  // namespace minimind
