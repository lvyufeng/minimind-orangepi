#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindAttentionTiling {
  uint32_t tokens;
  uint32_t metadata;
};

constexpr int32_t kMaxHeadDim = 128;
constexpr int32_t kMaxTokens = 2048;
constexpr float kNegativeInfinity = -3.4028234663852886e38F;

class KernelMiniMindAttention {
 public:
  __aicore__ inline KernelMiniMindAttention() = default;

  __aicore__ inline void Init(GM_ADDR query,
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
    keys_.SetGlobalBuffer((__gm__ half*)keys, tokens_ * kv_heads_ * head_dim_);
    values_.SetGlobalBuffer((__gm__ half*)values, tokens_ * kv_heads_ * head_dim_);
    out_.SetGlobalBuffer((__gm__ half*)out, q_heads_ * head_dim_);
    pipe_.InitBuffer(query_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(key_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(value_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(out_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(calc_buf_, (kMaxHeadDim * 6 + kMaxTokens) * sizeof(float));
  }

  __aicore__ inline void Process() {
    LocalTensor<float> work = calc_buf_.Get<float>();
    LocalTensor<float> output = work;
    LocalTensor<float> scores = work[kMaxHeadDim];
    LocalTensor<float> query_local = work[kMaxHeadDim + kMaxTokens];
    LocalTensor<float> key_local = query_local[kMaxHeadDim];
    LocalTensor<float> value_local = query_local[kMaxHeadDim * 2];
    LocalTensor<float> tmp = query_local[kMaxHeadDim * 3];
    LocalTensor<float> reduce = query_local[kMaxHeadDim * 4];

    const int64_t q_head = GetBlockIdx();
    if (q_head >= q_heads_) {
      return;
    }
    const int64_t kv_head = q_head / kv_repeat_;
    CopyQuery(query_local, q_head);
    const float inv_sqrt_dim = 1.0F / sqrt(static_cast<float>(head_dim_));
    float max_score = kNegativeInfinity;
    for (int64_t token = 0; token < tokens_; ++token) {
      CopyKey(key_local, token, kv_head);
      Mul(tmp, query_local, key_local, head_dim_);
      PipeBarrier<PIPE_V>();
      ReduceSum(reduce, tmp, value_local, static_cast<int32_t>(head_dim_));
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      const float score = reduce.GetValue(0) * inv_sqrt_dim;
      scores.SetValue(token, score);
      max_score = score > max_score ? score : max_score;
    }

    Adds(scores, scores, -max_score, tokens_);
    PipeBarrier<PIPE_V>();
    Exp(scores, scores, tokens_);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduce, scores, value_local, tokens_);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float denom = reduce.GetValue(0);

    Duplicate(output, 0.0F, head_dim_);
    PipeBarrier<PIPE_V>();
    for (int64_t token = 0; token < tokens_; ++token) {
      const float weight = scores.GetValue(token) / denom;
      CopyValue(value_local, token, kv_head);
      Muls(tmp, value_local, weight, head_dim_);
      PipeBarrier<PIPE_V>();
      Add(output, output, tmp, head_dim_);
      PipeBarrier<PIPE_V>();
    }
    CopyOut(output, q_head);
  }

 private:
  __aicore__ inline void CopyQuery(LocalTensor<float> query_local, int64_t q_head) {
    const int64_t offset = q_head * head_dim_;
    LocalTensor<half> query_half = query_queue_.AllocTensor<half>();
    DataCopy(query_half, query_[offset], head_dim_);
    query_queue_.EnQue(query_half);
    LocalTensor<half> ready = query_queue_.DeQue<half>();
    Cast(query_local, ready, RoundMode::CAST_NONE, head_dim_);
    query_queue_.FreeTensor(ready);
  }

  __aicore__ inline void CopyKey(LocalTensor<float> key_local, int64_t token, int64_t kv_head) {
    const int64_t offset = (token * kv_heads_ + kv_head) * head_dim_;
    LocalTensor<half> key_half = key_queue_.AllocTensor<half>();
    DataCopy(key_half, keys_[offset], head_dim_);
    key_queue_.EnQue(key_half);
    LocalTensor<half> ready = key_queue_.DeQue<half>();
    Cast(key_local, ready, RoundMode::CAST_NONE, head_dim_);
    key_queue_.FreeTensor(ready);
  }

  __aicore__ inline void CopyValue(LocalTensor<float> value_local, int64_t token, int64_t kv_head) {
    const int64_t offset = (token * kv_heads_ + kv_head) * head_dim_;
    LocalTensor<half> value_half = value_queue_.AllocTensor<half>();
    DataCopy(value_half, values_[offset], head_dim_);
    value_queue_.EnQue(value_half);
    LocalTensor<half> ready = value_queue_.DeQue<half>();
    Cast(value_local, ready, RoundMode::CAST_NONE, head_dim_);
    value_queue_.FreeTensor(ready);
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
  TQue<TPosition::VECIN, 1> query_queue_;
  TQue<TPosition::VECIN, 1> key_queue_;
  TQue<TPosition::VECIN, 1> value_queue_;
  TQue<TPosition::VECOUT, 1> out_queue_;
  TBuf<TPosition::VECCALC> calc_buf_;
  GlobalTensor<half> query_;
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

extern "C" __global__ __aicore__ void mini_mind_attention(GM_ADDR query,
                                                           GM_ADDR keys,
                                                           GM_ADDR values,
                                                           GM_ADDR out,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindAttentionTiling* tiling_data = reinterpret_cast<__gm__ MiniMindAttentionTiling*>(tiling);
  KernelMiniMindAttention op;
  op.Init(query, keys, values, out, tiling_data);
  op.Process();
}
