#include "../../src/csrc/models/minimind/ascend_custom_ops.h"

#include <cmath>
#include <iostream>
#include <vector>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  if (!minimind::model::custom_ops_available()) {
    return 0;
  }

  std::vector<float> input(128);
  std::vector<float> weight(128);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125F;
    weight[i] = 0.5F + static_cast<float>(i % 7) * 0.03125F;
  }

  const float eps = 1e-5F;
  const auto normalized = minimind::model::custom_rms_norm(input, weight, eps);
  CHECK(normalized.size() == input.size());

  float sum = 0.0F;
  for (float value : input) {
    sum += value * value;
  }
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(input.size()) + eps);
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float expected = input[i] * scale * weight[i];
    CHECK(std::fabs(normalized[i] - expected) < 5e-2F);
  }

  std::vector<float> gate(128);
  std::vector<float> up(128);
  for (std::size_t i = 0; i < gate.size(); ++i) {
    gate[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125F;
    up[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.0625F;
  }

  const auto activated = minimind::model::custom_swiglu(gate, up);
  CHECK(activated.size() == gate.size());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const float silu = gate[i] / (1.0F + std::exp(-gate[i]));
    CHECK(std::fabs(activated[i] - silu * up[i]) < 5e-2F);
  }

  return 0;
}
