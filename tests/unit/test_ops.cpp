#include "minimind_orangepi/ops.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

namespace {

bool close(float lhs, float rhs) {
  return std::fabs(lhs - rhs) < 1e-5F;
}

}  // namespace

int main() {
  minimind::HostTensor input(minimind::TensorShape({2, 2}), minimind::DType::kFloat32);
  input.data_as<float>()[0] = 3.0F;
  input.data_as<float>()[1] = 4.0F;
  input.data_as<float>()[2] = 1.0F;
  input.data_as<float>()[3] = 2.0F;
  minimind::HostTensor weight(minimind::TensorShape({2}), minimind::DType::kFloat32);
  weight.data_as<float>()[0] = 1.0F;
  weight.data_as<float>()[1] = 2.0F;

  auto normalized = minimind::rms_norm(input, weight, 0.0F);
  const float first_scale = 1.0F / std::sqrt((9.0F + 16.0F) / 2.0F);
  CHECK(close(normalized.data_as<float>()[0], 3.0F * first_scale));
  CHECK(close(normalized.data_as<float>()[1], 4.0F * first_scale * 2.0F));

  minimind::HostTensor gate(minimind::TensorShape({3}), minimind::DType::kFloat32);
  minimind::HostTensor up(minimind::TensorShape({3}), minimind::DType::kFloat32);
  gate.data_as<float>()[0] = 0.0F;
  gate.data_as<float>()[1] = 1.0F;
  gate.data_as<float>()[2] = -1.0F;
  up.data_as<float>()[0] = 2.0F;
  up.data_as<float>()[1] = 3.0F;
  up.data_as<float>()[2] = 4.0F;
  auto output = minimind::swiglu(gate, up);
  CHECK(close(output.data_as<float>()[0], 0.0F));
  CHECK(close(output.data_as<float>()[1], (1.0F / (1.0F + std::exp(-1.0F))) * 3.0F));
  CHECK(close(output.data_as<float>()[2], (-1.0F / (1.0F + std::exp(1.0F))) * 4.0F));

  bool threw = false;
  try {
    (void)minimind::swiglu(gate, input);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  return 0;
}
