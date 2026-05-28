#pragma once

#include <cstdint>
#include <string>

namespace minimind {

class AclContext {
 public:
  AclContext() = default;
  AclContext(const AclContext&) = delete;
  AclContext& operator=(const AclContext&) = delete;
  AclContext(AclContext&& other) noexcept;
  AclContext& operator=(AclContext&& other) noexcept;
  ~AclContext();

  void initialize(int32_t device_id = 0);
  void reset();

  bool initialized() const noexcept { return initialized_; }
  int32_t device_id() const noexcept { return device_id_; }
  const std::string& backend() const noexcept { return backend_; }

 private:
  bool initialized_ = false;
  int32_t device_id_ = -1;
  std::string backend_ = "host";
};

}  // namespace minimind
