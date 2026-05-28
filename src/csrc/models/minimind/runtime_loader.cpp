#include "runtime_loader.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace minimind::model {

namespace {

std::string trim(std::string value) {
  std::size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::unordered_map<std::string, std::string> read_config_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open runtime config: " + path.string());
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::runtime_error("invalid runtime config line: " + line);
    }
    values[trim(line.substr(0, equals))] = trim(line.substr(equals + 1));
  }
  return values;
}

int64_t get_i64(const std::unordered_map<std::string, std::string>& values,
                const std::string& key,
                int64_t fallback) {
  const auto it = values.find(key);
  if (it == values.end()) {
    return fallback;
  }
  return std::stoll(it->second);
}

bool get_bool(const std::unordered_map<std::string, std::string>& values,
              const std::string& key,
              bool fallback) {
  const auto it = values.find(key);
  if (it == values.end()) {
    return fallback;
  }
  return it->second == "1" || it->second == "true" || it->second == "True";
}

MiniMindConfig config_from_file(const std::filesystem::path& path) {
  const auto values = read_config_file(path);
  MiniMindConfig config = default_minimind_config();
  config.hidden_size = get_i64(values, "hidden_size", config.hidden_size);
  config.num_hidden_layers = get_i64(values, "num_hidden_layers", config.num_hidden_layers);
  config.use_moe = get_bool(values, "use_moe", false);
  config.vocab_size = get_i64(values, "vocab_size", config.vocab_size);
  config.num_attention_heads = get_i64(values, "num_attention_heads", config.num_attention_heads);
  config.num_key_value_heads = get_i64(values, "num_key_value_heads", config.num_key_value_heads);
  config.head_dim = get_i64(values, "head_dim", config.hidden_size / config.num_attention_heads);
  config.intermediate_size = get_i64(values, "intermediate_size", config.intermediate_size);
  config.max_position_embeddings = get_i64(values, "max_position_embeddings", config.max_position_embeddings);
  config.tie_word_embeddings = get_bool(values, "tie_word_embeddings", config.tie_word_embeddings);
  config.moe_intermediate_size = get_i64(values, "moe_intermediate_size", config.intermediate_size);
  return config;
}

void read_exact(std::ifstream& input, char* data, std::size_t bytes, const char* name) {
  input.read(data, static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error(std::string("failed to read ") + name);
  }
}

uint64_t read_u64(std::ifstream& input) {
  uint64_t value = 0;
  read_exact(input, reinterpret_cast<char*>(&value), sizeof(value), "uint64");
  return value;
}

std::string read_string(std::ifstream& input) {
  const uint64_t size = read_u64(input);
  std::string value(static_cast<std::size_t>(size), '\0');
  read_exact(input, value.data(), value.size(), "string");
  return value;
}

std::vector<float> read_tensor(std::ifstream& input, const std::string& expected_name) {
  const std::string name = read_string(input);
  if (name != expected_name) {
    throw std::runtime_error("expected tensor " + expected_name + ", got " + name);
  }
  const uint64_t size = read_u64(input);
  std::vector<float> values(static_cast<std::size_t>(size));
  read_exact(input, reinterpret_cast<char*>(values.data()), values.size() * sizeof(float), expected_name.c_str());
  return values;
}

DenseModelWeights read_weights(const MiniMindConfig& config, const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open runtime weights: " + path.string());
  }
  char magic[8]{};
  read_exact(input, magic, sizeof(magic), "magic");
  if (std::string(magic, sizeof(magic)) != "MMRTW001") {
    throw std::runtime_error("unsupported runtime weight format");
  }

  DenseModelWeights weights;
  weights.embed_tokens = read_tensor(input, "model.embed_tokens.weight");
  weights.final_norm = read_tensor(input, "model.norm.weight");
  weights.lm_head = read_tensor(input, "lm_head.weight");
  weights.layers.reserve(static_cast<std::size_t>(config.num_hidden_layers));
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    DenseLayerWeights layer_weights;
    layer_weights.input_norm = read_tensor(input, prefix + "input_layernorm.weight");
    layer_weights.post_attention_norm = read_tensor(input, prefix + "post_attention_layernorm.weight");
    layer_weights.q_norm = read_tensor(input, prefix + "self_attn.q_norm.weight");
    layer_weights.k_norm = read_tensor(input, prefix + "self_attn.k_norm.weight");
    layer_weights.q_proj = read_tensor(input, prefix + "self_attn.q_proj.weight");
    layer_weights.k_proj = read_tensor(input, prefix + "self_attn.k_proj.weight");
    layer_weights.v_proj = read_tensor(input, prefix + "self_attn.v_proj.weight");
    layer_weights.o_proj = read_tensor(input, prefix + "self_attn.o_proj.weight");
    layer_weights.gate_proj = read_tensor(input, prefix + "mlp.gate_proj.weight");
    layer_weights.up_proj = read_tensor(input, prefix + "mlp.up_proj.weight");
    layer_weights.down_proj = read_tensor(input, prefix + "mlp.down_proj.weight");
    weights.layers.push_back(std::move(layer_weights));
  }
  return weights;
}

}  // namespace

bool has_runtime_language_model(const std::string& model_dir) {
  const std::filesystem::path dir(model_dir);
  return std::filesystem::exists(dir / "minimind_runtime_config.txt") &&
         std::filesystem::exists(dir / "weights.bin");
}

LanguageModel load_runtime_language_model(const std::string& model_dir) {
  const std::filesystem::path dir(model_dir);
  MiniMindConfig config = config_from_file(dir / "minimind_runtime_config.txt");
  DenseModelWeights weights = read_weights(config, dir / "weights.bin");
  return LanguageModel(std::move(config), std::move(weights));
}

}  // namespace minimind::model
