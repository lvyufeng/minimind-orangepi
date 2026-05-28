#include "projector.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace minimind::model_v {

namespace {

void require_size(const std::vector<float>& values, int64_t expected, const char* name) {
  if (static_cast<int64_t>(values.size()) != expected) {
    throw std::invalid_argument(std::string(name) + " has invalid size");
  }
}

float gelu(float value) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  return 0.5F * value * (1.0F + std::tanh(kSqrtTwoOverPi * (value + 0.044715F * value * value * value)));
}

std::vector<float> matvec(const std::vector<float>& matrix,
                          const std::vector<float>& bias,
                          int64_t rows,
                          int64_t cols,
                          const float* input) {
  require_size(matrix, rows * cols, "projector matrix");
  require_size(bias, rows, "projector bias");
  std::vector<float> output(static_cast<std::size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    float sum = bias[static_cast<std::size_t>(row)];
    for (int64_t col = 0; col < cols; ++col) {
      sum += matrix[static_cast<std::size_t>(row * cols + col)] * input[col];
    }
    output[static_cast<std::size_t>(row)] = sum;
  }
  return output;
}

std::vector<float> layer_norm(const float* input,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              int64_t size) {
  require_size(weight, size, "projector norm_weight");
  require_size(bias, size, "projector norm_bias");
  float mean = 0.0F;
  for (int64_t i = 0; i < size; ++i) {
    mean += input[i];
  }
  mean /= static_cast<float>(size);

  float variance = 0.0F;
  for (int64_t i = 0; i < size; ++i) {
    const float centered = input[i] - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(size);
  const float scale = 1.0F / std::sqrt(variance + 1e-5F);

  std::vector<float> output(static_cast<std::size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    output[static_cast<std::size_t>(i)] = (input[i] - mean) * scale * weight[static_cast<std::size_t>(i)] + bias[static_cast<std::size_t>(i)];
  }
  return output;
}

std::vector<float> identity_matrix(int64_t rows, int64_t cols) {
  std::vector<float> matrix(static_cast<std::size_t>(rows * cols), 0.0F);
  for (int64_t i = 0; i < std::min(rows, cols); ++i) {
    matrix[static_cast<std::size_t>(i * cols + i)] = 1.0F;
  }
  return matrix;
}

}  // namespace

std::vector<float> run_projector(const MiniMindVConfig& config,
                                 const ProjectorWeights& weights,
                                 const std::vector<float>& image_features) {
  const int64_t tokens = config.image_token_len;
  const int64_t in_dim = config.image_hidden_size;
  const int64_t out_dim = config.text.hidden_size;
  require_size(image_features, tokens * in_dim, "image_features");

  std::vector<float> output(static_cast<std::size_t>(tokens * out_dim));
  for (int64_t token = 0; token < tokens; ++token) {
    const float* input = image_features.data() + token * in_dim;
    auto normed = layer_norm(input, weights.norm_weight, weights.norm_bias, in_dim);
    auto hidden = matvec(weights.fc1_weight, weights.fc1_bias, out_dim, in_dim, normed.data());
    for (float& value : hidden) {
      value = gelu(value);
    }
    auto projected = matvec(weights.fc2_weight, weights.fc2_bias, out_dim, out_dim, hidden.data());
    std::copy(projected.begin(), projected.end(), output.begin() + token * out_dim);
  }
  return output;
}

ProjectorWeights make_identity_projector_weights(const MiniMindVConfig& config) {
  ProjectorWeights weights;
  weights.norm_weight.assign(static_cast<std::size_t>(config.image_hidden_size), 1.0F);
  weights.norm_bias.assign(static_cast<std::size_t>(config.image_hidden_size), 0.0F);
  weights.fc1_weight = identity_matrix(config.text.hidden_size, config.image_hidden_size);
  weights.fc1_bias.assign(static_cast<std::size_t>(config.text.hidden_size), 0.0F);
  weights.fc2_weight = identity_matrix(config.text.hidden_size, config.text.hidden_size);
  weights.fc2_bias.assign(static_cast<std::size_t>(config.text.hidden_size), 0.0F);
  return weights;
}

}  // namespace minimind::model_v
