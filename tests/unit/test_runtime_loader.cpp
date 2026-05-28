#include "../../src/csrc/models/minimind/language_model.h"
#include "../../src/csrc/models/minimind/runtime_loader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

namespace {

void write_u64(std::ofstream& output, uint64_t value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_tensor(std::ofstream& output, const std::string& name, const std::vector<float>& values) {
  write_u64(output, name.size());
  output.write(name.data(), static_cast<std::streamsize>(name.size()));
  write_u64(output, values.size());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<float> filled(int64_t size, float value) {
  return std::vector<float>(static_cast<std::size_t>(size), value);
}

std::vector<float> identity(int64_t rows, int64_t cols) {
  std::vector<float> matrix(static_cast<std::size_t>(rows * cols), 0.0F);
  for (int64_t i = 0; i < std::min(rows, cols); ++i) {
    matrix[static_cast<std::size_t>(i * cols + i)] = 1.0F;
  }
  return matrix;
}

}  // namespace

int main() {
  const auto dir = std::filesystem::temp_directory_path() / "minimind_runtime_loader_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  {
    std::ofstream config(dir / "minimind_runtime_config.txt");
    config << "hidden_size=4\n"
           << "num_hidden_layers=1\n"
           << "use_moe=0\n"
           << "vocab_size=8\n"
           << "num_attention_heads=2\n"
           << "num_key_value_heads=1\n"
           << "head_dim=2\n"
           << "intermediate_size=8\n"
           << "tie_word_embeddings=0\n";
  }

  {
    std::ofstream weights(dir / "weights.bin", std::ios::binary);
    weights.write("MMRTW001", 8);
    write_tensor(weights, "model.embed_tokens.weight", filled(8 * 4, 0.1F));
    write_tensor(weights, "model.norm.weight", filled(4, 1.0F));
    write_tensor(weights, "lm_head.weight", filled(8 * 4, 0.2F));
    write_tensor(weights, "model.layers.0.input_layernorm.weight", filled(4, 1.0F));
    write_tensor(weights, "model.layers.0.post_attention_layernorm.weight", filled(4, 1.0F));
    write_tensor(weights, "model.layers.0.self_attn.q_norm.weight", filled(2, 1.0F));
    write_tensor(weights, "model.layers.0.self_attn.k_norm.weight", filled(2, 1.0F));
    write_tensor(weights, "model.layers.0.self_attn.q_proj.weight", identity(4, 4));
    write_tensor(weights, "model.layers.0.self_attn.k_proj.weight", identity(2, 4));
    write_tensor(weights, "model.layers.0.self_attn.v_proj.weight", identity(2, 4));
    write_tensor(weights, "model.layers.0.self_attn.o_proj.weight", identity(4, 4));
    write_tensor(weights, "model.layers.0.mlp.gate_proj.weight", identity(8, 4));
    write_tensor(weights, "model.layers.0.mlp.up_proj.weight", identity(8, 4));
    write_tensor(weights, "model.layers.0.mlp.down_proj.weight", identity(4, 8));
  }

  CHECK(minimind::model::has_runtime_language_model(dir.string()));
  auto model = minimind::model::load_runtime_language_model(dir.string());
  CHECK(model.config().hidden_size == 4);
  CHECK(model.config().vocab_size == 8);
  const auto generated = model.generate({1}, 2);
  CHECK(!generated.empty());

  std::filesystem::remove_all(dir);
  return 0;
}
