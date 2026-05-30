#include "ascend_custom_ops.h"
#include "ascend_matmul.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_embedding.h>
#include <aclnnop/aclnn_slice.h>
#include <aclnn_mini_mind_add.h>
#include <aclnn_mini_mind_attention.h>
#include <aclnn_mini_mind_decode_qkv_postprocess.h>
#include <aclnn_mini_mind_decode_attention.h>
#include <aclnn_mini_mind_prefill_attention.h>
#include <aclnn_mini_mind_qk_norm_rope.h>
#include <aclnn_mini_mind_qkv_split_rows.h>
#include <aclnn_mini_mind_rms_norm.h>
#include <aclnn_mini_mind_rms_norm_rows.h>
#include <aclnn_mini_mind_rope.h>
#include <aclnn_mini_mind_swi_glu.h>
#include <aclnn_mini_mind_swi_glu_rows.h>
#endif

namespace minimind::model {

namespace {

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)

std::vector<uint16_t> to_half_vector(const std::vector<float>& values) {
  std::vector<uint16_t> half_values(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    half_values[i] = float_to_half(values[i]);
  }
  return half_values;
}

DeviceBuffer copy_half_to_device(const std::vector<float>& values) {
  const auto half_values = to_half_vector(values);

  DeviceBuffer device(half_values.size() * sizeof(uint16_t));
  check_acl(aclrtMemcpy(device.data(), device.bytes(), half_values.data(), device.bytes(), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy H2D failed");
  return device;
}

}  // namespace

struct CustomAttentionCache {
  int64_t capacity_tokens = 0;
  int64_t kv_heads = 0;
  int64_t head_dim = 0;
  DeviceBuffer keys;
  DeviceBuffer values;
};

namespace {

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
struct RopeTableCacheKey {
  int64_t max_positions = 0;
  int64_t head_dim = 0;
  float theta = 0.0F;

  bool operator<(const RopeTableCacheKey& other) const {
    if (max_positions != other.max_positions) {
      return max_positions < other.max_positions;
    }
    if (head_dim != other.head_dim) {
      return head_dim < other.head_dim;
    }
    return theta < other.theta;
  }
};

struct RopeTableCacheEntry {
  DeviceBuffer cos;
  DeviceBuffer sin;
  int64_t max_positions = 0;
  int64_t half_dim = 0;
};

RopeTableCacheEntry& cached_rope_table(int64_t max_positions, int64_t head_dim, float theta) {
  static std::mutex* mutex = new std::mutex();
  static std::map<RopeTableCacheKey, RopeTableCacheEntry>* cache = new std::map<RopeTableCacheKey, RopeTableCacheEntry>();
  const RopeTableCacheKey key{max_positions, head_dim, theta};
  std::lock_guard<std::mutex> lock(*mutex);
  auto it = cache->find(key);
  if (it != cache->end()) {
    return it->second;
  }

  const int64_t half_dim = head_dim / 2;
  std::vector<float> cos_values(static_cast<std::size_t>(max_positions * half_dim));
  std::vector<float> sin_values(static_cast<std::size_t>(max_positions * half_dim));
  for (int64_t position = 0; position < max_positions; ++position) {
    for (int64_t dim = 0; dim < half_dim; ++dim) {
      const float inv_freq = 1.0F / std::pow(theta, static_cast<float>(dim * 2) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position) * inv_freq;
      const std::size_t index = static_cast<std::size_t>(position * half_dim + dim);
      cos_values[index] = std::cos(angle);
      sin_values[index] = std::sin(angle);
    }
  }

  RopeTableCacheEntry entry;
  entry.cos = copy_half_to_device(cos_values);
  entry.sin = copy_half_to_device(sin_values);
  entry.max_positions = max_positions;
  entry.half_dim = half_dim;
  auto [inserted, _] = cache->emplace(key, std::move(entry));
  return inserted->second;
}
#endif

void ensure_attention_cache(CustomAttentionCache& cache, int64_t tokens, int64_t kv_heads, int64_t head_dim) {
  if (tokens <= 0 || kv_heads <= 0 || head_dim <= 0) {
    throw std::invalid_argument("invalid attention cache shape");
  }
  if (cache.capacity_tokens >= tokens && cache.kv_heads == kv_heads && cache.head_dim == head_dim) {
    return;
  }
  cache.capacity_tokens = std::max<int64_t>(tokens, 2048);
  cache.kv_heads = kv_heads;
  cache.head_dim = head_dim;
  const std::size_t bytes = static_cast<std::size_t>(cache.capacity_tokens * kv_heads * head_dim) * sizeof(uint16_t);
  cache.keys.reset(bytes);
  cache.values.reset(bytes);
}

DeviceBuffer& custom_workspace() {
  static DeviceBuffer* workspace = new DeviceBuffer();
  return *workspace;
}

std::vector<float> copy_half_to_host(const DeviceBuffer& device, std::size_t values) {
  std::vector<uint16_t> half_values(values);
  check_acl(aclrtMemcpy(half_values.data(), half_values.size() * sizeof(uint16_t), device.data(), device.bytes(),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy D2H failed");

  std::vector<float> output(values);
  for (std::size_t i = 0; i < values; ++i) {
    output[i] = half_to_float(half_values[i]);
  }
  return output;
}


#endif

}  // namespace

bool custom_ops_available() {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  return cube_matvec_available();
#else
  return false;
#endif
}

std::vector<float> custom_add(const std::vector<float>& lhs,
                              const std::vector<float>& rhs) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("custom_add shape mismatch");
  }
  (void)runtime();

  DeviceBuffer lhs_device = copy_half_to_device(lhs);
  DeviceBuffer rhs_device = copy_half_to_device(rhs);
  DeviceBuffer output_device(lhs.size() * sizeof(uint16_t));

  const int64_t dims[1] = {static_cast<int64_t>(lhs.size())};
  const int64_t strides[1] = {1};
  TensorHandle lhs_tensor(dims, strides, 1, lhs_device.data());
  TensorHandle rhs_tensor(dims, strides, 1, rhs_device.data());
  TensorHandle out(dims, strides, 1, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindAddGetWorkspaceSize(lhs_tensor.get(), rhs_tensor.get(), out.get(), &workspace_size, &executor),
              "aclnnMiniMindAddGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindAdd(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindAdd failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, lhs.size());
#else
  (void)lhs;
  (void)rhs;
  throw std::runtime_error("custom_add is unavailable");
#endif
}

std::vector<float> custom_rms_norm(const std::vector<float>& input,
                                   const std::vector<float>& weight,
                                   float eps) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (input.size() != weight.size()) {
    throw std::invalid_argument("custom_rms_norm shape mismatch");
  }
  (void)runtime();

  DeviceBuffer input_device = copy_half_to_device(input);
  DeviceBuffer weight_device = copy_half_to_device(weight);
  DeviceBuffer output_device(input.size() * sizeof(uint16_t));

  const int64_t dims[1] = {static_cast<int64_t>(input.size())};
  const int64_t strides[1] = {1};
  TensorHandle x(dims, strides, 1, input_device.data());
  TensorHandle gamma(dims, strides, 1, weight_device.data());
  TensorHandle out(dims, strides, 1, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindRmsNormGetWorkspaceSize(x.get(), gamma.get(), static_cast<double>(eps), out.get(),
                                                   &workspace_size, &executor),
              "aclnnMiniMindRmsNormGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindRmsNorm(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindRmsNorm failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, input.size());
#else
  (void)eps;
  throw std::runtime_error("custom_rms_norm is unavailable");
#endif
}

std::vector<float> custom_swiglu(const std::vector<float>& gate,
                                 const std::vector<float>& up) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (gate.size() != up.size()) {
    throw std::invalid_argument("custom_swiglu shape mismatch");
  }
  (void)runtime();

  DeviceBuffer gate_device = copy_half_to_device(gate);
  DeviceBuffer up_device = copy_half_to_device(up);
  DeviceBuffer output_device(gate.size() * sizeof(uint16_t));

  const int64_t dims[1] = {static_cast<int64_t>(gate.size())};
  const int64_t strides[1] = {1};
  TensorHandle gate_tensor(dims, strides, 1, gate_device.data());
  TensorHandle up_tensor(dims, strides, 1, up_device.data());
  TensorHandle out(dims, strides, 1, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindSwiGluGetWorkspaceSize(gate_tensor.get(), up_tensor.get(), out.get(),
                                                  &workspace_size, &executor),
              "aclnnMiniMindSwiGluGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindSwiGlu(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindSwiGlu failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, gate.size());
#else
  throw std::runtime_error("custom_swiglu is unavailable");
#endif
}

std::vector<float> custom_rope(const std::vector<float>& input,
                               int64_t heads,
                               int64_t head_dim,
                               int64_t position,
                               float theta) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 || static_cast<int64_t>(input.size()) != heads * head_dim) {
    throw std::invalid_argument("custom_rope shape mismatch");
  }
  (void)runtime();

  const int64_t half_dim = head_dim / 2;
  std::vector<float> cos_table(static_cast<std::size_t>(half_dim));
  std::vector<float> sin_table(static_cast<std::size_t>(half_dim));
  for (int64_t dim = 0; dim < half_dim; ++dim) {
    const float inv_freq = 1.0F / std::pow(theta, static_cast<float>(dim * 2) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * inv_freq;
    cos_table[static_cast<std::size_t>(dim)] = std::cos(angle);
    sin_table[static_cast<std::size_t>(dim)] = std::sin(angle);
  }

  DeviceBuffer input_device = copy_half_to_device(input);
  DeviceBuffer cos_device = copy_half_to_device(cos_table);
  DeviceBuffer sin_device = copy_half_to_device(sin_table);
  DeviceBuffer output_device(input.size() * sizeof(uint16_t));

  const int64_t dims[2] = {heads, head_dim};
  const int64_t strides[2] = {head_dim, 1};
  const int64_t table_dims[1] = {half_dim};
  const int64_t table_strides[1] = {1};
  TensorHandle x(dims, strides, 2, input_device.data());
  TensorHandle cos_tensor(table_dims, table_strides, 1, cos_device.data());
  TensorHandle sin_tensor(table_dims, table_strides, 1, sin_device.data());
  TensorHandle out(dims, strides, 2, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindRopeGetWorkspaceSize(x.get(), cos_tensor.get(), sin_tensor.get(), out.get(),
                                                &workspace_size, &executor),
              "aclnnMiniMindRopeGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindRope(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindRope failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, input.size());
#else
  (void)input;
  (void)heads;
  (void)head_dim;
  (void)position;
  (void)theta;
  throw std::runtime_error("custom_rope is unavailable");
#endif
}

std::vector<float> custom_attention(const std::vector<float>& query,
                                    const std::vector<float>& keys,
                                    const std::vector<float>& values,
                                    int64_t tokens,
                                    int64_t q_heads,
                                    int64_t kv_heads,
                                    int64_t head_dim) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || q_heads % kv_heads != 0) {
    throw std::invalid_argument("custom_attention invalid shape");
  }
  if (head_dim > 128 || tokens > 2048 || q_heads > 255 || kv_heads > 255 ||
      static_cast<int64_t>(query.size()) != q_heads * head_dim ||
      static_cast<int64_t>(keys.size()) != tokens * kv_heads * head_dim ||
      static_cast<int64_t>(values.size()) != tokens * kv_heads * head_dim) {
    throw std::invalid_argument("custom_attention shape mismatch");
  }
  (void)runtime();

  DeviceBuffer query_device = copy_half_to_device(query);
  DeviceBuffer keys_device = copy_half_to_device(keys);
  DeviceBuffer values_device = copy_half_to_device(values);
  DeviceBuffer output_device(query.size() * sizeof(uint16_t));

  const int64_t query_dims[2] = {q_heads, head_dim};
  const int64_t query_strides[2] = {head_dim, 1};
  const int64_t cache_dims[3] = {tokens, kv_heads, head_dim};
  const int64_t cache_strides[3] = {kv_heads * head_dim, head_dim, 1};
  TensorHandle query_tensor(query_dims, query_strides, 2, query_device.data());
  TensorHandle keys_tensor(cache_dims, cache_strides, 3, keys_device.data());
  TensorHandle values_tensor(cache_dims, cache_strides, 3, values_device.data());
  TensorHandle out(query_dims, query_strides, 2, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindAttentionGetWorkspaceSize(query_tensor.get(), keys_tensor.get(), values_tensor.get(), out.get(),
                                                     &workspace_size, &executor),
              "aclnnMiniMindAttentionGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindAttention(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindAttention failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, query.size());
#else
  (void)query;
  (void)keys;
  (void)values;
  (void)tokens;
  (void)q_heads;
  (void)kv_heads;
  (void)head_dim;
  throw std::runtime_error("custom_attention is unavailable");
#endif
}

std::vector<float> custom_prefill_attention(const std::vector<float>& query,
                                            const std::vector<float>& keys,
                                            const std::vector<float>& values,
                                            std::shared_ptr<CustomAttentionCache>& cache,
                                            int64_t seq_len,
                                            int64_t q_heads,
                                            int64_t kv_heads,
                                            int64_t head_dim) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (seq_len <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || q_heads % kv_heads != 0) {
    throw std::invalid_argument("custom_prefill_attention invalid shape");
  }
  if (head_dim > 128 || seq_len > 2048 || q_heads > 255 || kv_heads > 255 ||
      static_cast<int64_t>(query.size()) != seq_len * q_heads * head_dim ||
      static_cast<int64_t>(keys.size()) != seq_len * kv_heads * head_dim ||
      static_cast<int64_t>(values.size()) != seq_len * kv_heads * head_dim) {
    throw std::invalid_argument("custom_prefill_attention shape mismatch");
  }
  (void)runtime();

  if (!cache || cache->kv_heads != kv_heads || cache->head_dim != head_dim || cache->capacity_tokens < seq_len) {
    cache = std::make_shared<CustomAttentionCache>();
    ensure_attention_cache(*cache, seq_len, kv_heads, head_dim);
  }
  ensure_attention_cache(*cache, seq_len, kv_heads, head_dim);

  DeviceBuffer query_device = copy_half_to_device(query);
  DeviceBuffer output_device(query.size() * sizeof(uint16_t));

  const auto key_half = to_half_vector(keys);
  const auto value_half = to_half_vector(values);
  const std::size_t cache_bytes = key_half.size() * sizeof(uint16_t);
  check_acl(aclrtMemcpy(cache->keys.data(), cache->keys.bytes(), key_half.data(), cache_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy prefill keys H2D failed");
  check_acl(aclrtMemcpy(cache->values.data(), cache->values.bytes(), value_half.data(), cache_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy prefill values H2D failed");

  const int64_t query_dims[3] = {seq_len, q_heads, head_dim};
  const int64_t query_strides[3] = {q_heads * head_dim, head_dim, 1};
  const int64_t cache_dims[3] = {seq_len, kv_heads, head_dim};
  const int64_t cache_strides[3] = {kv_heads * head_dim, head_dim, 1};
  TensorHandle query_tensor(query_dims, query_strides, 3, query_device.data());
  TensorHandle keys_tensor(cache_dims, cache_strides, 3, cache->keys.data());
  TensorHandle values_tensor(cache_dims, cache_strides, 3, cache->values.data());
  TensorHandle out(query_dims, query_strides, 3, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindPrefillAttentionGetWorkspaceSize(query_tensor.get(), keys_tensor.get(), values_tensor.get(),
                                                            out.get(), &workspace_size, &executor),
              "aclnnMiniMindPrefillAttentionGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindPrefillAttention(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindPrefillAttention failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, query.size());
#else
  (void)query;
  (void)keys;
  (void)values;
  (void)cache;
  (void)seq_len;
  (void)q_heads;
  (void)kv_heads;
  (void)head_dim;
  throw std::runtime_error("custom_prefill_attention is unavailable");
#endif
}

std::vector<float> custom_attention_cached(const std::vector<float>& query,
                                           const std::vector<float>& key,
                                           const std::vector<float>& value,
                                           std::shared_ptr<CustomAttentionCache>& cache,
                                           int64_t tokens,
                                           int64_t q_heads,
                                           int64_t kv_heads,
                                           int64_t head_dim) {
#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || q_heads % kv_heads != 0) {
    throw std::invalid_argument("custom_attention_cached invalid shape");
  }
  if (head_dim > 128 || tokens > 2048 || q_heads > 255 || kv_heads > 255 ||
      static_cast<int64_t>(query.size()) != q_heads * head_dim ||
      static_cast<int64_t>(key.size()) != kv_heads * head_dim ||
      static_cast<int64_t>(value.size()) != kv_heads * head_dim) {
    throw std::invalid_argument("custom_attention_cached shape mismatch");
  }
  (void)runtime();

  if (!cache || cache->kv_heads != kv_heads || cache->head_dim != head_dim || cache->capacity_tokens < tokens) {
    cache = std::make_shared<CustomAttentionCache>();
    ensure_attention_cache(*cache, tokens, kv_heads, head_dim);
  }
  ensure_attention_cache(*cache, tokens, kv_heads, head_dim);

  const int64_t kv_size = kv_heads * head_dim;
  const auto key_half = to_half_vector(key);
  const auto value_half = to_half_vector(value);
  const std::size_t bytes = key_half.size() * sizeof(uint16_t);
  const std::size_t offset = static_cast<std::size_t>((tokens - 1) * kv_size) * sizeof(uint16_t);
  auto* key_target = static_cast<unsigned char*>(cache->keys.data()) + offset;
  auto* value_target = static_cast<unsigned char*>(cache->values.data()) + offset;
  check_acl(aclrtMemcpy(key_target, cache->keys.bytes() - offset, key_half.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy cached key H2D failed");
  check_acl(aclrtMemcpy(value_target, cache->values.bytes() - offset, value_half.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy cached value H2D failed");

  DeviceBuffer query_device = copy_half_to_device(query);
  DeviceBuffer output_device(query.size() * sizeof(uint16_t));

  const int64_t query_dims[2] = {q_heads, head_dim};
  const int64_t query_strides[2] = {head_dim, 1};
  const int64_t cache_dims[3] = {tokens, kv_heads, head_dim};
  const int64_t cache_strides[3] = {kv_heads * head_dim, head_dim, 1};
  TensorHandle query_tensor(query_dims, query_strides, 2, query_device.data());
  TensorHandle keys_tensor(cache_dims, cache_strides, 3, cache->keys.data());
  TensorHandle values_tensor(cache_dims, cache_strides, 3, cache->values.data());
  TensorHandle out(query_dims, query_strides, 2, output_device.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindAttentionGetWorkspaceSize(query_tensor.get(), keys_tensor.get(), values_tensor.get(), out.get(),
                                                     &workspace_size, &executor),
              "aclnnMiniMindAttentionGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindAttention(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindAttention failed");
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream failed");
  return copy_half_to_host(output_device, query.size());
#else
  (void)query;
  (void)key;
  (void)value;
  (void)cache;
  (void)tokens;
  (void)q_heads;
  (void)kv_heads;
  (void)head_dim;
  throw std::runtime_error("custom_attention_cached is unavailable");
#endif
}

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)

void device_add(const DeviceTensor& lhs, const DeviceTensor& rhs, DeviceTensor& out) {
  if (lhs.elements() != rhs.elements() || lhs.elements() != out.elements()) {
    throw std::invalid_argument("device_add shape mismatch");
  }
  const int64_t dims[1] = {lhs.elements()};
  const int64_t strides[1] = {1};
  TensorHandle lhs_tensor(dims, strides, 1, lhs.data());
  TensorHandle rhs_tensor(dims, strides, 1, rhs.data());
  TensorHandle out_tensor(dims, strides, 1, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindAddGetWorkspaceSize(lhs_tensor.get(), rhs_tensor.get(), out_tensor.get(),
                                               &workspace_size, &executor),
              "aclnnMiniMindAddGetWorkspaceSize device failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindAdd(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindAdd device failed");
}

void device_rms_norm_rows(const DeviceTensor& input,
                          const DeviceTensor& weight,
                          float eps,
                          int64_t rows,
                          int64_t cols,
                          DeviceTensor& out) {
  if (input.elements() != rows * cols || out.elements() != rows * cols || weight.elements() != cols) {
    throw std::invalid_argument("device_rms_norm_rows shape mismatch");
  }
  const int64_t input_dims[2] = {rows, cols};
  const int64_t input_strides[2] = {cols, 1};
  const int64_t weight_dims[1] = {cols};
  const int64_t weight_strides[1] = {1};
  TensorHandle x(input_dims, input_strides, 2, input.data());
  TensorHandle gamma(weight_dims, weight_strides, 1, weight.data());
  TensorHandle y(input_dims, input_strides, 2, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindRmsNormRowsGetWorkspaceSize(x.get(), gamma.get(), static_cast<double>(eps), y.get(),
                                                       &workspace_size, &executor),
              "aclnnMiniMindRmsNormRowsGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindRmsNormRows(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindRmsNormRows failed");
}

void device_swiglu_rows(const DeviceTensor& gate_up, int64_t rows, int64_t cols, DeviceTensor& out) {
  if (gate_up.elements() != rows * cols * 2 || out.elements() != rows * cols) {
    throw std::invalid_argument("device_swiglu_rows shape mismatch");
  }
  const int64_t gate_up_dims[2] = {rows, cols * 2};
  const int64_t gate_up_strides[2] = {cols * 2, 1};
  const int64_t out_dims[2] = {rows, cols};
  const int64_t out_strides[2] = {cols, 1};
  TensorHandle gate_up_tensor(gate_up_dims, gate_up_strides, 2, gate_up.data());
  TensorHandle out_tensor(out_dims, out_strides, 2, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindSwiGluRowsGetWorkspaceSize(gate_up_tensor.get(), out_tensor.get(),
                                                      &workspace_size, &executor),
              "aclnnMiniMindSwiGluRowsGetWorkspaceSize device failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindSwiGluRows(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindSwiGluRows device failed");
}

void device_qk_norm_rope(DeviceTensor& query,
                         DeviceTensor& key,
                         const DeviceTensor& q_weight,
                         const DeviceTensor& k_weight,
                         int64_t q_heads,
                         int64_t kv_heads,
                         int64_t head_dim,
                         int64_t position,
                         float theta) {
  if (query.elements() != q_heads * head_dim || key.elements() != kv_heads * head_dim ||
      q_weight.elements() != head_dim || k_weight.elements() != head_dim || head_dim % 2 != 0) {
    throw std::invalid_argument("device_qk_norm_rope shape mismatch");
  }
  RopeTableCacheEntry& table = cached_rope_table(32768, head_dim, theta);
  const int64_t query_dims[2] = {q_heads, head_dim};
  const int64_t query_strides[2] = {head_dim, 1};
  const int64_t key_dims[2] = {kv_heads, head_dim};
  const int64_t key_strides[2] = {head_dim, 1};
  const int64_t weight_dims[1] = {head_dim};
  const int64_t weight_strides[1] = {1};
  const int64_t table_dims[1] = {head_dim / 2};
  const int64_t table_strides[1] = {1};
  auto* cos_data = static_cast<unsigned char*>(table.cos.data()) +
                   static_cast<std::size_t>(position * table.half_dim) * sizeof(uint16_t);
  auto* sin_data = static_cast<unsigned char*>(table.sin.data()) +
                   static_cast<std::size_t>(position * table.half_dim) * sizeof(uint16_t);
  TensorHandle query_tensor(query_dims, query_strides, 2, query.data());
  TensorHandle key_tensor(key_dims, key_strides, 2, key.data());
  TensorHandle q_weight_tensor(weight_dims, weight_strides, 1, q_weight.data());
  TensorHandle k_weight_tensor(weight_dims, weight_strides, 1, k_weight.data());
  TensorHandle cos_tensor(table_dims, table_strides, 1, cos_data);
  TensorHandle sin_tensor(table_dims, table_strides, 1, sin_data);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindQkNormRopeGetWorkspaceSize(query_tensor.get(), key_tensor.get(), q_weight_tensor.get(),
                                                      k_weight_tensor.get(), cos_tensor.get(), sin_tensor.get(),
                                                      &workspace_size, &executor),
              "aclnnMiniMindQkNormRopeGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindQkNormRope(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindQkNormRope failed");
}

void device_qkv_split_rows(const DeviceTensor& qkv,
                           int64_t rows,
                           int64_t q_size,
                           int64_t kv_size,
                           DeviceTensor& query,
                           DeviceTensor& key,
                           DeviceTensor& value) {
  const int64_t row_size = q_size + kv_size * 2;
  if (qkv.elements() != rows * row_size || query.elements() != rows * q_size ||
      key.elements() != rows * kv_size || value.elements() != rows * kv_size) {
    throw std::invalid_argument("device_qkv_split_rows shape mismatch");
  }
  const int64_t qkv_dims[2] = {rows, row_size};
  const int64_t qkv_strides[2] = {row_size, 1};
  const int64_t query_dims[2] = {rows, q_size};
  const int64_t query_strides[2] = {q_size, 1};
  const int64_t kv_dims[2] = {rows, kv_size};
  const int64_t kv_strides[2] = {kv_size, 1};
  TensorHandle qkv_tensor(qkv_dims, qkv_strides, 2, qkv.data());
  TensorHandle query_tensor(query_dims, query_strides, 2, query.data());
  TensorHandle key_tensor(kv_dims, kv_strides, 2, key.data());
  TensorHandle value_tensor(kv_dims, kv_strides, 2, value.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindQkvSplitRowsGetWorkspaceSize(qkv_tensor.get(), q_size, kv_size, query_tensor.get(),
                                                        key_tensor.get(), value_tensor.get(), &workspace_size,
                                                        &executor),
              "aclnnMiniMindQkvSplitRowsGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindQkvSplitRows(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindQkvSplitRows failed");
}

void device_copy_last_row(const DeviceTensor& input, int64_t rows, int64_t cols, DeviceTensor& out) {
  if (input.elements() != rows * cols || out.elements() != cols || rows <= 0 || cols <= 0) {
    throw std::invalid_argument("device_copy_last_row shape mismatch");
  }
  const std::size_t bytes = static_cast<std::size_t>(cols) * sizeof(uint16_t);
  auto* source = static_cast<unsigned char*>(input.data()) + static_cast<std::size_t>((rows - 1) * cols) * sizeof(uint16_t);
  check_acl(aclrtMemcpyAsync(out.data(), out.bytes(), source, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, runtime().stream()),
            "aclrtMemcpyAsync device_copy_last_row D2D failed");
}

void device_rope_rows(DeviceTensor& tensor,
                      int64_t rows,
                      int64_t heads,
                      int64_t head_dim,
                      int64_t start_position,
                      int64_t max_positions,
                      float theta) {
  if (tensor.elements() != rows * heads * head_dim || head_dim % 2 != 0 || start_position < 0 ||
      start_position + rows > max_positions) {
    throw std::invalid_argument("device_rope_rows shape mismatch");
  }
  const int64_t half_dim = head_dim / 2;
  RopeTableCacheEntry& table = cached_rope_table(max_positions, head_dim, theta);
  DeviceBuffer& workspace = custom_workspace();
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t dims[2] = {heads, head_dim};
    const int64_t strides[2] = {head_dim, 1};
    const int64_t table_dims[1] = {half_dim};
    const int64_t table_strides[1] = {1};
    auto* row_data = static_cast<unsigned char*>(tensor.data()) +
                     static_cast<std::size_t>(row * heads * head_dim) * sizeof(uint16_t);
    auto* cos_data = static_cast<unsigned char*>(table.cos.data()) +
                     static_cast<std::size_t>((start_position + row) * half_dim) * sizeof(uint16_t);
    auto* sin_data = static_cast<unsigned char*>(table.sin.data()) +
                     static_cast<std::size_t>((start_position + row) * half_dim) * sizeof(uint16_t);
    TensorHandle x(dims, strides, 2, row_data);
    TensorHandle cos_tensor(table_dims, table_strides, 1, cos_data);
    TensorHandle sin_tensor(table_dims, table_strides, 1, sin_data);
    TensorHandle out(dims, strides, 2, row_data);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    check_aclnn(aclnnMiniMindRopeGetWorkspaceSize(x.get(), cos_tensor.get(), sin_tensor.get(), out.get(),
                                                  &workspace_size, &executor),
                "aclnnMiniMindRopeGetWorkspaceSize device failed");
    workspace.reserve(workspace_size);
    check_aclnn(aclnnMiniMindRope(workspace.data(), workspace_size, executor, runtime().stream()),
                "aclnnMiniMindRope device failed");
  }
}

void device_embedding_gather(const DeviceTensor& weight,
                             const DeviceTensor& token_ids,
                             int64_t tokens,
                             int64_t vocab_size,
                             int64_t hidden_size,
                             DeviceTensor& out) {
  if (tokens * hidden_size != out.elements() || weight.elements() != vocab_size * hidden_size ||
      token_ids.elements() != tokens || token_ids.dtype != ACL_INT32) {
    throw std::invalid_argument("device_embedding_gather shape mismatch");
  }

  const int64_t weight_dims[2] = {vocab_size, hidden_size};
  const int64_t weight_strides[2] = {hidden_size, 1};
  const int64_t index_dims[1] = {tokens};
  const int64_t index_strides[1] = {1};
  const int64_t out_dims[2] = {tokens, hidden_size};
  const int64_t out_strides[2] = {hidden_size, 1};
  TensorHandle weight_tensor(weight_dims, weight_strides, 2, weight.data());
  TensorHandle indices_tensor(index_dims, index_strides, 1, token_ids.data(), ACL_INT32);
  TensorHandle out_tensor(out_dims, out_strides, 2, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnEmbeddingGetWorkspaceSize(weight_tensor.get(), indices_tensor.get(), out_tensor.get(),
                                             &workspace_size, &executor),
              "aclnnEmbeddingGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnEmbedding(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnEmbedding failed");
}

void device_embedding_gather(const DeviceTensor& weight,
                             const std::vector<int32_t>& token_ids,
                             int64_t vocab_size,
                             int64_t hidden_size,
                             DeviceTensor& out) {
  DeviceTensor indices = upload_tensor_int32(token_ids, {static_cast<int64_t>(token_ids.size())});
  device_embedding_gather(weight, indices, static_cast<int64_t>(token_ids.size()), vocab_size, hidden_size, out);
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream device_embedding_gather failed");
}

void device_prefill_attention(const DeviceTensor& query,
                              const DeviceTensor& keys,
                              const DeviceTensor& values,
                              std::shared_ptr<CustomAttentionCache>& cache,
                              int64_t seq_len,
                              int64_t q_heads,
                              int64_t kv_heads,
                              int64_t head_dim,
                              DeviceTensor& out) {
  if (query.elements() != seq_len * q_heads * head_dim || keys.elements() != seq_len * kv_heads * head_dim ||
      values.elements() != seq_len * kv_heads * head_dim || out.elements() != query.elements()) {
    throw std::invalid_argument("device_prefill_attention shape mismatch");
  }
  if (!cache || cache->kv_heads != kv_heads || cache->head_dim != head_dim || cache->capacity_tokens < seq_len) {
    cache = std::make_shared<CustomAttentionCache>();
    ensure_attention_cache(*cache, seq_len, kv_heads, head_dim);
  }
  ensure_attention_cache(*cache, seq_len, kv_heads, head_dim);
  const std::size_t cache_bytes = static_cast<std::size_t>(seq_len * kv_heads * head_dim) * sizeof(uint16_t);
  check_acl(aclrtMemcpyAsync(cache->keys.data(), cache->keys.bytes(), keys.data(), cache_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                             runtime().stream()),
            "aclrtMemcpyAsync prefill keys D2D failed");
  check_acl(aclrtMemcpyAsync(cache->values.data(), cache->values.bytes(), values.data(), cache_bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, runtime().stream()),
            "aclrtMemcpyAsync prefill values D2D failed");

  const int64_t query_dims[3] = {seq_len, q_heads, head_dim};
  const int64_t query_strides[3] = {q_heads * head_dim, head_dim, 1};
  const int64_t cache_dims[3] = {seq_len, kv_heads, head_dim};
  const int64_t cache_strides[3] = {kv_heads * head_dim, head_dim, 1};
  TensorHandle query_tensor(query_dims, query_strides, 3, query.data());
  TensorHandle keys_tensor(cache_dims, cache_strides, 3, cache->keys.data());
  TensorHandle values_tensor(cache_dims, cache_strides, 3, cache->values.data());
  TensorHandle out_tensor(query_dims, query_strides, 3, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindPrefillAttentionGetWorkspaceSize(query_tensor.get(), keys_tensor.get(), values_tensor.get(),
                                                            out_tensor.get(), &workspace_size, &executor),
              "aclnnMiniMindPrefillAttentionGetWorkspaceSize device failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindPrefillAttention(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindPrefillAttention device failed");
}

void device_decode_qkv_postprocess(const DeviceTensor& qkv,
                                   const DeviceTensor& q_weight,
                                   const DeviceTensor& k_weight,
                                   std::shared_ptr<CustomAttentionCache>& cache,
                                   int64_t tokens,
                                   int64_t q_heads,
                                   int64_t kv_heads,
                                   int64_t head_dim,
                                   int64_t position,
                                   int64_t max_positions,
                                   float theta,
                                   DeviceTensor& query) {
  const int64_t q_size = q_heads * head_dim;
  const int64_t kv_size = kv_heads * head_dim;
  if (qkv.elements() != q_size + 2 * kv_size || q_weight.elements() != head_dim || k_weight.elements() != head_dim ||
      query.elements() != q_size || head_dim % 2 != 0 || position < 0 || position >= max_positions) {
    throw std::invalid_argument("device_decode_qkv_postprocess shape mismatch");
  }
  if (!cache || cache->kv_heads != kv_heads || cache->head_dim != head_dim || cache->capacity_tokens < tokens) {
    cache = std::make_shared<CustomAttentionCache>();
    ensure_attention_cache(*cache, tokens, kv_heads, head_dim);
  }
  ensure_attention_cache(*cache, tokens, kv_heads, head_dim);
  RopeTableCacheEntry& table = cached_rope_table(max_positions, head_dim, theta);
  auto* cos_data = static_cast<unsigned char*>(table.cos.data()) +
                   static_cast<std::size_t>(position * table.half_dim) * sizeof(uint16_t);
  auto* sin_data = static_cast<unsigned char*>(table.sin.data()) +
                   static_cast<std::size_t>(position * table.half_dim) * sizeof(uint16_t);

  const int64_t qkv_dims[2] = {1, q_size + 2 * kv_size};
  const int64_t qkv_strides[2] = {q_size + 2 * kv_size, 1};
  const int64_t weight_dims[1] = {head_dim};
  const int64_t weight_strides[1] = {1};
  const int64_t table_dims[1] = {head_dim / 2};
  const int64_t table_strides[1] = {1};
  const int64_t cache_dims[3] = {cache->capacity_tokens, kv_heads, head_dim};
  const int64_t cache_strides[3] = {kv_heads * head_dim, head_dim, 1};
  const int64_t query_dims[2] = {q_heads, head_dim};
  const int64_t query_strides[2] = {head_dim, 1};
  TensorHandle qkv_tensor(qkv_dims, qkv_strides, 2, qkv.data());
  TensorHandle q_weight_tensor(weight_dims, weight_strides, 1, q_weight.data());
  TensorHandle k_weight_tensor(weight_dims, weight_strides, 1, k_weight.data());
  TensorHandle cos_tensor(table_dims, table_strides, 1, cos_data);
  TensorHandle sin_tensor(table_dims, table_strides, 1, sin_data);
  TensorHandle keys_tensor(cache_dims, cache_strides, 3, cache->keys.data());
  TensorHandle values_tensor(cache_dims, cache_strides, 3, cache->values.data());
  TensorHandle query_tensor(query_dims, query_strides, 2, query.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindDecodeQkvPostprocessGetWorkspaceSize(
                  qkv_tensor.get(), q_weight_tensor.get(), k_weight_tensor.get(), cos_tensor.get(), sin_tensor.get(),
                  keys_tensor.get(), values_tensor.get(), q_heads, kv_heads, head_dim, tokens, query_tensor.get(),
                  &workspace_size, &executor),
              "aclnnMiniMindDecodeQkvPostprocessGetWorkspaceSize failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindDecodeQkvPostprocess(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindDecodeQkvPostprocess failed");
}

void device_attention_from_cache(const DeviceTensor& query,
                                 const std::shared_ptr<CustomAttentionCache>& cache,
                                 int64_t tokens,
                                 int64_t q_heads,
                                 int64_t kv_heads,
                                 int64_t head_dim,
                                 DeviceTensor& out) {
  if (!cache || query.elements() != q_heads * head_dim || out.elements() != q_heads * head_dim || tokens <= 0 ||
      cache->kv_heads != kv_heads || cache->head_dim != head_dim || cache->capacity_tokens < tokens) {
    throw std::invalid_argument("device_attention_from_cache shape mismatch");
  }

  const int64_t query_dims[2] = {q_heads, head_dim};
  const int64_t query_strides[2] = {head_dim, 1};
  const int64_t cache_dims[3] = {tokens, kv_heads, head_dim};
  const int64_t cache_strides[3] = {kv_heads * head_dim, head_dim, 1};
  TensorHandle query_tensor(query_dims, query_strides, 2, query.data());
  TensorHandle keys_tensor(cache_dims, cache_strides, 3, cache->keys.data());
  TensorHandle values_tensor(cache_dims, cache_strides, 3, cache->values.data());
  TensorHandle out_tensor(query_dims, query_strides, 2, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindAttentionGetWorkspaceSize(query_tensor.get(), keys_tensor.get(), values_tensor.get(),
                                                     out_tensor.get(), &workspace_size, &executor),
              "aclnnMiniMindAttentionGetWorkspaceSize device failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindAttention(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindAttention device failed");
}

void device_decode_attention(const DeviceTensor& query,
                             const DeviceTensor& key,
                             const DeviceTensor& value,
                             std::shared_ptr<CustomAttentionCache>& cache,
                             int64_t tokens,
                             int64_t q_heads,
                             int64_t kv_heads,
                             int64_t head_dim,
                             DeviceTensor& out) {
  if (query.elements() != q_heads * head_dim || key.elements() != kv_heads * head_dim ||
      value.elements() != kv_heads * head_dim || out.elements() != q_heads * head_dim) {
    throw std::invalid_argument("device_decode_attention shape mismatch");
  }
  if (!cache || cache->kv_heads != kv_heads || cache->head_dim != head_dim || cache->capacity_tokens < tokens) {
    cache = std::make_shared<CustomAttentionCache>();
    ensure_attention_cache(*cache, tokens, kv_heads, head_dim);
  }
  ensure_attention_cache(*cache, tokens, kv_heads, head_dim);
  const int64_t kv_size = kv_heads * head_dim;
  const std::size_t bytes = static_cast<std::size_t>(kv_size) * sizeof(uint16_t);
  const std::size_t offset = static_cast<std::size_t>((tokens - 1) * kv_size) * sizeof(uint16_t);
  auto* key_target = static_cast<unsigned char*>(cache->keys.data()) + offset;
  auto* value_target = static_cast<unsigned char*>(cache->values.data()) + offset;
  check_acl(aclrtMemcpyAsync(key_target, cache->keys.bytes() - offset, key.data(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                             runtime().stream()),
            "aclrtMemcpyAsync decode key D2D failed");
  check_acl(aclrtMemcpyAsync(value_target, cache->values.bytes() - offset, value.data(), bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, runtime().stream()),
            "aclrtMemcpyAsync decode value D2D failed");

  const int64_t query_dims[2] = {q_heads, head_dim};
  const int64_t query_strides[2] = {head_dim, 1};
  const int64_t cache_dims[3] = {tokens, kv_heads, head_dim};
  const int64_t cache_strides[3] = {kv_heads * head_dim, head_dim, 1};
  TensorHandle query_tensor(query_dims, query_strides, 2, query.data());
  TensorHandle keys_tensor(cache_dims, cache_strides, 3, cache->keys.data());
  TensorHandle values_tensor(cache_dims, cache_strides, 3, cache->values.data());
  TensorHandle out_tensor(query_dims, query_strides, 2, out.data());

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(aclnnMiniMindAttentionGetWorkspaceSize(query_tensor.get(), keys_tensor.get(), values_tensor.get(),
                                                     out_tensor.get(), &workspace_size, &executor),
              "aclnnMiniMindAttentionGetWorkspaceSize device failed");
  DeviceBuffer& workspace = custom_workspace();
  workspace.reserve(workspace_size);
  check_aclnn(aclnnMiniMindAttention(workspace.data(), workspace_size, executor, runtime().stream()),
              "aclnnMiniMindAttention device failed");
}

#endif

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)

std::vector<float> custom_rms_norm_rows(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         int64_t rows,
                                         int64_t cols,
                                         float eps) {
  if (static_cast<int64_t>(input.size()) != rows * cols || static_cast<int64_t>(weight.size()) != cols) {
    throw std::invalid_argument("custom_rms_norm_rows shape mismatch");
  }
  DeviceTensor input_device = upload_tensor_fp16(input, {rows, cols});
  DeviceTensor weight_device = upload_tensor_fp16(weight, {cols});
  DeviceTensor output_device = make_device_tensor({rows, cols});
  device_rms_norm_rows(input_device, weight_device, eps, rows, cols, output_device);
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream custom_rms_norm_rows failed");
  std::vector<float> output;
  copy_tensor_to_host_fp16(output_device, output);
  return output;
}

std::vector<float> custom_swiglu_rows(const std::vector<float>& gate_up,
                                      int64_t rows,
                                      int64_t cols) {
  if (static_cast<int64_t>(gate_up.size()) != rows * cols * 2) {
    throw std::invalid_argument("custom_swiglu_rows shape mismatch");
  }
  DeviceTensor input_device = upload_tensor_fp16(gate_up, {rows, cols * 2});
  DeviceTensor output_device = make_device_tensor({rows, cols});
  device_swiglu_rows(input_device, rows, cols, output_device);
  check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream custom_swiglu_rows failed");
  std::vector<float> output;
  copy_tensor_to_host_fp16(output_device, output);
  return output;
}

#endif

}  // namespace minimind::model
