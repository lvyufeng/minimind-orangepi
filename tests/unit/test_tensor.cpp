#include "minimind_orangepi/ops.h"
#include "minimind_orangepi/tensor.h"

#include <iostream>
#include <stdexcept>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  minimind::TensorShape shape({2, 3, 4});
  CHECK(shape.rank() == 3);
  CHECK(shape.dim(1) == 3);
  CHECK(shape.numel() == 24);
  CHECK(shape.str() == "[2, 3, 4]");
  CHECK(minimind::dtype_name(minimind::DType::kFloat16) == "float16");
  CHECK(minimind::dtype_size(minimind::DType::kInt32) == 4);

  minimind::HostTensor lhs(minimind::TensorShape({3}), minimind::DType::kFloat32);
  minimind::HostTensor rhs(minimind::TensorShape({3}), minimind::DType::kFloat32);
  lhs.data_as<float>()[0] = 1.0F;
  lhs.data_as<float>()[1] = 2.0F;
  lhs.data_as<float>()[2] = 3.0F;
  rhs.data_as<float>()[0] = 4.0F;
  rhs.data_as<float>()[1] = 5.0F;
  rhs.data_as<float>()[2] = 6.0F;

  auto sum = minimind::add(lhs, rhs);
  CHECK(sum.data_as<float>()[0] == 5.0F);
  CHECK(sum.data_as<float>()[1] == 7.0F);
  CHECK(sum.data_as<float>()[2] == 9.0F);

  bool threw = false;
  try {
    minimind::TensorShape bad({1, -1});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  return 0;
}
