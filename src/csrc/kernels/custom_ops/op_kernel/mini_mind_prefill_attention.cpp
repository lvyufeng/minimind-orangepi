#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindPrefillAttentionTiling {
  uint32_t seq_len;
  uint32_t metadata;
};

constexpr int32_t kMaxHeadDim = 128;
constexpr int32_t kMaxTokens = 2048;
constexpr float kNegativeInfinity = -3.4028234663852886e38F;

class KernelMiniMindPrefillAttention {
 public:
  __aicore__ inline KernelMiniMindPrefillAttention() = default;

  __aicore__ inline void Init(GM_ADDR query,
                              GM_ADDR keys,
                              GM_ADDR values,
                              GM_ADDR out,
                              const __gm__ MiniMindPrefillAttentionTiling* tiling) {
    seq_len_ = tiling->seq_len;
    head_dim_ = tiling->metadata & 0xFFFFU;
    kv_heads_ = (tiling->metadata >> 16U) & 0xFFU;
    q_heads_ = (tiling->metadata >> 24U) & 0xFFU;
    kv_repeat_ = q_heads_ / kv_heads_;
    const int64_t q_total = seq_len_ * q_heads_ * head_dim_;
    const int64_t kv_total = seq_len_ * kv_heads_ * head_dim_;
    query_.SetGlobalBuffer((__gm__ half*)query, q_total);
    keys_.SetGlobalBuffer((__gm__ half*)keys, kv_total);
    values_.SetGlobalBuffer((__gm__ half*)values, kv_total);
    out_.SetGlobalBuffer((__gm__ half*)out, q_total);
    pipe_.InitBuffer(query_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(key_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(value_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(out_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(calc_buf_, (kMaxHeadDim * 6 + kMaxTokens) * sizeof(float));
  }

  __aicore__ inline void Process() {
    if (head_dim_ < 8) {
      ProcessScalar();
      return;
    }

    LocalTensor<float> work = calc_buf_.Get<float>();
    LocalTensor<float> output = work;
    LocalTensor<float> scores = work[kMaxHeadDim];
    LocalTensor<float> query_local = work[kMaxHeadDim + kMaxTokens];
    LocalTensor<float> key_local = query_local[kMaxHeadDim];
    LocalTensor<float> value_local = query_local[kMaxHeadDim * 2];
    LocalTensor<float> tmp = query_local[kMaxHeadDim * 3];
    LocalTensor<float> reduce = query_local[kMaxHeadDim * 4];
    const float inv_sqrt_dim = 1.0F / sqrt(static_cast<float>(head_dim_));

    for (int64_t pos = 0; pos < seq_len_; ++pos) {
      const int64_t valid_tokens = pos + 1;
      for (int64_t q_head = 0; q_head < q_heads_; ++q_head) {
        const int64_t kv_head = q_head / kv_repeat_;
        CopyQuery(query_local, pos, q_head);
        float max_score = kNegativeInfinity;
        for (int64_t token = 0; token < valid_tokens; ++token) {
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

        Adds(scores, scores, -max_score, valid_tokens);
        PipeBarrier<PIPE_V>();
        Exp(scores, scores, valid_tokens);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduce, scores, value_local, valid_tokens);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float denom = reduce.GetValue(0);

        Duplicate(output, 0.0F, head_dim_);
        PipeBarrier<PIPE_V>();
        for (int64_t token = 0; token < valid_tokens; ++token) {
          const float weight = scores.GetValue(token) / denom;
          CopyValue(value_local, token, kv_head);
          Muls(tmp, value_local, weight, head_dim_);
          PipeBarrier<PIPE_V>();
          Add(output, output, tmp, head_dim_);
          PipeBarrier<PIPE_V>();
        }
        CopyOut(output, pos, q_head);
      }
    }
  }

 private:
  __aicore__ inline void ProcessScalar() {
    LocalTensor<float> work = calc_buf_.Get<float>();
    LocalTensor<float> output = work;
    LocalTensor<float> scores = work[kMaxHeadDim];
    LocalTensor<float> scratch = work[kMaxHeadDim + kMaxTokens];
    LocalTensor<float> query_local = scratch;
    LocalTensor<float> key_local = scratch[kMaxHeadDim];
    LocalTensor<float> value_local = scratch[kMaxHeadDim * 2];

    for (int64_t pos = 0; pos < seq_len_; ++pos) {
      const int64_t valid_tokens = pos + 1;
      for (int64_t q_head = 0; q_head < q_heads_; ++q_head) {
        const int64_t kv_head = q_head / kv_repeat_;
        CopyQueryScalar(query_local, pos, q_head);
        float max_score = kNegativeInfinity;
        for (int64_t token = 0; token < valid_tokens; ++token) {
          CopyKeyScalar(key_local, token, kv_head);
          float dot = 0.0F;
          for (int64_t dim = 0; dim < head_dim_; ++dim) {
            dot += query_local.GetValue(dim) * key_local.GetValue(dim);
          }
          const float score = dot / sqrt(static_cast<float>(head_dim_));
          scores.SetValue(token, score);
          max_score = score > max_score ? score : max_score;
        }

        Adds(scores, scores, -max_score, valid_tokens);
        PipeBarrier<PIPE_V>();
        Exp(scores, scores, valid_tokens);
        PipeBarrier<PIPE_V>();
        ReduceSum(value_local, scores, query_local, valid_tokens);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float denom = value_local.GetValue(0);

        for (int64_t dim = 0; dim < head_dim_; ++dim) {
          output.SetValue(dim, 0.0F);
        }
        for (int64_t token = 0; token < valid_tokens; ++token) {
          const float weight = scores.GetValue(token) / denom;
          CopyValueScalar(value_local, token, kv_head);
          for (int64_t dim = 0; dim < head_dim_; ++dim) {
            output.SetValue(dim, output.GetValue(dim) + weight * value_local.GetValue(dim));
          }
        }
        CopyOut(output, pos, q_head);
      }
    }
  }

  __aicore__ inline void CopyQuery(LocalTensor<float> query_local, int64_t pos, int64_t q_head) {
    const int64_t offset = (pos * q_heads_ + q_head) * head_dim_;
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

  __aicore__ inline void CopyQueryScalar(LocalTensor<float> query_local, int64_t pos, int64_t q_head) {
    const int64_t offset = (pos * q_heads_ + q_head) * head_dim_;
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      query_local.SetValue(dim, static_cast<float>(query_.GetValue(offset + dim)));
    }
  }

  __aicore__ inline void CopyKeyScalar(LocalTensor<float> key_local, int64_t token, int64_t kv_head) {
    const int64_t offset = (token * kv_heads_ + kv_head) * head_dim_;
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      key_local.SetValue(dim, static_cast<float>(keys_.GetValue(offset + dim)));
    }
  }

  __aicore__ inline void CopyValueScalar(LocalTensor<float> value_local, int64_t token, int64_t kv_head) {
    const int64_t offset = (token * kv_heads_ + kv_head) * head_dim_;
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      value_local.SetValue(dim, static_cast<float>(values_.GetValue(offset + dim)));
    }
  }

  __aicore__ inline void CopyOut(LocalTensor<float> output, int64_t pos, int64_t q_head) {
    LocalTensor<half> out_local = out_queue_.AllocTensor<half>();
    Cast(out_local, output, RoundMode::CAST_NONE, head_dim_);
    out_queue_.EnQue(out_local);
    LocalTensor<half> ready = out_queue_.DeQue<half>();
    DataCopy(out_[(pos * q_heads_ + q_head) * head_dim_], ready, head_dim_);
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
  int64_t seq_len_ = 0;
  int64_t q_heads_ = 0;
  int64_t kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t kv_repeat_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_prefill_attention(GM_ADDR query,
                                                                  GM_ADDR keys,
                                                                  GM_ADDR values,
                                                                  GM_ADDR out,
                                                                  GM_ADDR workspace,
                                                                  GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindPrefillAttentionTiling* tiling_data =
      reinterpret_cast<__gm__ MiniMindPrefillAttentionTiling*>(tiling);
  KernelMiniMindPrefillAttention op;
  op.Init(query, keys, values, out, tiling_data);
  op.Process();
}
