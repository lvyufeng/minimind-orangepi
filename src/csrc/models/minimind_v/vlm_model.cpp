#include "vlm_model.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace minimind::model_v {

namespace {

void require_size(const std::vector<float>& values, int64_t expected, const char* name) {
  if (static_cast<int64_t>(values.size()) != expected) {
    throw std::invalid_argument(std::string(name) + " has invalid size");
  }
}

}  // namespace

std::vector<ImageTokenSpan> find_image_token_spans(const std::vector<int32_t>& input_ids,
                                                   int32_t image_token_id) {
  std::vector<ImageTokenSpan> spans;
  int64_t index = 0;
  while (index < static_cast<int64_t>(input_ids.size())) {
    if (input_ids[static_cast<std::size_t>(index)] != image_token_id) {
      ++index;
      continue;
    }
    const int64_t start = index;
    while (index < static_cast<int64_t>(input_ids.size()) &&
           input_ids[static_cast<std::size_t>(index)] == image_token_id) {
      ++index;
    }
    spans.push_back(ImageTokenSpan{start, index - start});
  }
  return spans;
}

std::vector<float> replace_image_token_embeddings(
    const MiniMindVConfig& config,
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& token_embeddings,
    const std::vector<float>& image_embeddings) {
  const int64_t hidden_size = config.text.hidden_size;
  require_size(token_embeddings, static_cast<int64_t>(input_ids.size()) * hidden_size, "token_embeddings");
  require_size(image_embeddings, config.image_token_len * hidden_size, "image_embeddings");

  const auto spans = find_image_token_spans(input_ids, config.image_token_id);
  if (spans.size() != 1) {
    throw std::invalid_argument("MiniMind-V baseline expects exactly one image token span");
  }
  if (spans.front().length != config.image_token_len) {
    throw std::invalid_argument("image token span length must equal image_token_len");
  }

  std::vector<float> output = token_embeddings;
  const int64_t start = spans.front().start;
  std::copy(image_embeddings.begin(), image_embeddings.end(),
            output.begin() + start * hidden_size);
  return output;
}

}  // namespace minimind::model_v
