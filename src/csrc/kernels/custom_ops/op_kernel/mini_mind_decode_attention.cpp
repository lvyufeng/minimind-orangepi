#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindAttentionTiling {
  uint32_t tokens;
  uint32_t metadata;
};

constexpr int32_t kBufferNum = 2;
constexpr int32_t kCopyTileLength = 256;
constexpr int32_t kMaxHeadDim = 128;
constexpr int32_t kMaxTokens = 2048;
constexpr float kNegativeInfinity = -3.4028234663852886e38F;

class KernelMiniMindDecodeAttention {
 public:
  __aicore__ inline KernelMiniMindDecodeAttention() = default;

  __aicore__ inline void Init(GM_ADDR query,
                              GM_ADDR key,
                              GM_ADDR value,
                              GM_ADDR keys,
                              GM_ADDR values,
                              GM_ADDR out,
                              const __gm__ MiniMindAttentionTiling* tiling) {
    tokens_ = tiling->tokens;
    head_dim_ = tiling->metadata & 0xFFFFU;
    kv_heads_ = (tiling->metadata >> 16U) & 0xFFU;
    q_heads_ = (tiling->metadata >> 24U) & 0xFFU;
    kv_repeat_ = q_heads_ / kv_heads_;
    query_.SetGlobalBuffer((__gm__ half*)query, q_heads_ * head_dim_);
    key_.SetGlobalBuffer((__gm__ half*)key, kv_heads_ * head_dim_);
    value_.SetGlobalBuffer((__gm__ half*)value, kv_heads_ * head_dim_);
    keys_.SetGlobalBuffer((__gm__ half*)keys, tokens_ * kv_heads_ * head_dim_);
    values_.SetGlobalBuffer((__gm__ half*)values, tokens_ * kv_heads_ * head_dim_);
    out_.SetGlobalBuffer((__gm__ half*)out, q_heads_ * head_dim_);
    pipe_.InitBuffer(copy_queue_, kBufferNum, kCopyTileLength * sizeof(half));
    pipe_.InitBuffer(out_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(calc_buf_, (kMaxHeadDim * 4 + kMaxTokens) * sizeof(float));
  }

  __aicore__ inline void Process() {
    const int64_t q_head = GetBlockIdx();
    if (q_head == 0) {
      WriteCurrentKvToCache();
    }
    if (q_head >= q_heads_) {
      return;
    }

    LocalTensor<float> work = calc_buf_.Get<float>();
    LocalTensor<float> output = work;
    LocalTensor<float> scores = work[kMaxHeadDim];
    LocalTensor<float> scratch = work[kMaxHeadDim + kMaxTokens];
    LocalTensor<float> query_local = scratch;
    LocalTensor<float> key_local = scratch[kMaxHeadDim];
    LocalTensor<float> value_local = scratch[kMaxHeadDim * 2];

    const int64_t kv_head = q_head / kv_repeat_;
    CopyQuery(query_local, q_head);
    float max_score = kNegativeInfinity;
    for (int64_t token = 0; token < tokens_; ++token) {
      CopyKey(key_local, token, kv_head);
      float dot = 0.0F;
      for (int64_t dim = 0; dim < head_dim_; ++dim) {
        dot += query_local.GetValue(dim) * key_local.GetValue(dim);
      }
      const float score = dot / sqrt(static_cast<float>(head_dim_));
      scores.SetValue(token, score);
      max_score = score > max_score ? score : max_score;
    }

    Adds(scores, scores, -max_score, tokens_);
    PipeBarrier<PIPE_V>();
    Exp(scores, scores, tokens_);
    PipeBarrier<PIPE_V>();
    ReduceSum(value_local, scores, query_local, tokens_);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float denom = value_local.GetValue(0);

    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      output.SetValue(dim, 0.0F);
    }
    for (int64_t token = 0; token < tokens_; ++token) {
      const float weight = scores.GetValue(token) / denom;
      CopyValue(value_local, token, kv_head);
      for (int64_t dim = 0; dim < head_dim_; ++dim) {
        output.SetValue(dim, output.GetValue(dim) + weight * value_local.GetValue(dim));
      }
    }
    CopyOut(output, q_head);
  }

 private:
  __aicore__ inline void WriteCurrentKvToCache() {
    const int64_t length = kv_heads_ * head_dim_;
    const int64_t cache_offset = (tokens_ - 1) * length;
    CopySegmentToCache(key_, keys_, cache_offset, length);
    CopySegmentToCache(value_, values_, cache_offset, length);
  }

  __aicore__ inline void CopySegmentToCache(GlobalTensor<half>& source,
                                            GlobalTensor<half>& target,
                                            int64_t target_offset,
                                            int64_t length) {
    for (int64_t offset = 0; offset < length; offset += kCopyTileLength) {
      const int64_t remaining = length - offset;
      const int64_t copy_length = kCopyTileLength < remaining ? kCopyTileLength : remaining;
      LocalTensor<half> local = copy_queue_.AllocTensor<half>();
      DataCopy(local, source[offset], copy_length);
      copy_queue_.EnQue(local);
      LocalTensor<half> ready = copy_queue_.DeQue<half>();
      DataCopy(target[target_offset + offset], ready, copy_length);
      copy_queue_.FreeTensor(ready);
    }
  }

  __aicore__ inline void CopyQuery(LocalTensor<float> query_local, int64_t q_head) {
    const int64_t offset = q_head * head_dim_;
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      query_local.SetValue(dim, static_cast<float>(query_.GetValue(offset + dim)));
    }
  }

  __aicore__ inline void CopyKey(LocalTensor<float> key_local, int64_t token, int64_t kv_head) {
    const int64_t offset = (token * kv_heads_ + kv_head) * head_dim_;
    if (token + 1 == tokens_) {
      const int64_t current_offset = kv_head * head_dim_;
      for (int64_t dim = 0; dim < head_dim_; ++dim) {
        key_local.SetValue(dim, static_cast<float>(key_.GetValue(current_offset + dim)));
      }
      return;
    }
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      key_local.SetValue(dim, static_cast<float>(keys_.GetValue(offset + dim)));
    }
  }

  __aicore__ inline void CopyValue(LocalTensor<float> value_local, int64_t token, int64_t kv_head) {
    const int64_t offset = (token * kv_heads_ + kv_head) * head_dim_;
    if (token + 1 == tokens_) {
      const int64_t current_offset = kv_head * head_dim_;
      for (int64_t dim = 0; dim < head_dim_; ++dim) {
        value_local.SetValue(dim, static_cast<float>(value_.GetValue(current_offset + dim)));
      }
      return;
    }
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      value_local.SetValue(dim, static_cast<float>(values_.GetValue(offset + dim)));
    }
  }

  __aicore__ inline void CopyOut(LocalTensor<float> output, int64_t q_head) {
    LocalTensor<half> out_local = out_queue_.AllocTensor<half>();
    Cast(out_local, output, RoundMode::CAST_NONE, head_dim_);
    out_queue_.EnQue(out_local);
    LocalTensor<half> ready = out_queue_.DeQue<half>();
    DataCopy(out_[q_head * head_dim_], ready, head_dim_);
    out_queue_.FreeTensor(ready);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, kBufferNum> copy_queue_;
  TQue<TPosition::VECOUT, 1> out_queue_;
  TBuf<TPosition::VECCALC> calc_buf_;
  GlobalTensor<half> query_;
  GlobalTensor<half> key_;
  GlobalTensor<half> value_;
  GlobalTensor<half> keys_;
  GlobalTensor<half> values_;
  GlobalTensor<half> out_;
  int64_t tokens_ = 0;
  int64_t q_heads_ = 0;
  int64_t kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t kv_repeat_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_decode_attention(GM_ADDR query,
                                                                  GM_ADDR key,
                                                                  GM_ADDR value,
                                                                  GM_ADDR keys,
                                                                  GM_ADDR values,
                                                                  GM_ADDR out,
                                                                  GM_ADDR workspace,
                                                                  GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindAttentionTiling* tiling_data = reinterpret_cast<__gm__ MiniMindAttentionTiling*>(tiling);
  KernelMiniMindDecodeAttention op;
  op.Init(query, key, value, keys, values, out, tiling_data);
  op.Process();
}
