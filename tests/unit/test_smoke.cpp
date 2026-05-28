#include "minimind_orangepi/acl_context.h"
#include "minimind_orangepi/ops.h"
#include "minimind_orangepi/tensor.h"

#include <iostream>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  minimind::AclContext context;
  context.initialize(0);
  CHECK(context.initialized());
  CHECK(context.device_id() == 0);
  CHECK(!context.backend().empty());
  context.reset();
  CHECK(!context.initialized());

  auto tensor = minimind::zeros(minimind::TensorShape({2, 2}), minimind::DType::kFloat32);
  CHECK(tensor.numel() == 4);
  CHECK(tensor.bytes() == 16);

  std::cout << "smoke ok\n";
  return 0;
}
