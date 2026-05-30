#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindQkNormRopeTiling {
  uint32_t metadata;
  uint32_t position;
};

constexpr int32_t kMaxHeadDim = 128;

class KernelMiniMindQkNormRope {
 public:
  __aicore__ inline KernelMiniMindQkNormRope() = default;

  __aicore__ inline void Init(GM_ADDR query,
                              GM_ADDR key,
                              GM_ADDR q_weight,
                              GM_ADDR k_weight,
                              GM_ADDR cos,
                              GM_ADDR sin,
                              const __gm__ MiniMindQkNormRopeTiling* tiling) {
    (void)tiling;
    head_dim_ = tiling->metadata & 0xFFU;
    kv_heads_ = (tiling->metadata >> 8U) & 0xFFU;
    q_heads_ = (tiling->metadata >> 16U) & 0xFFU;
    half_dim_ = head_dim_ / 2;
    q_total_ = q_heads_ * head_dim_;
    k_total_ = kv_heads_ * head_dim_;
    query_.SetGlobalBuffer((__gm__ half*)query, q_total_);
    key_.SetGlobalBuffer((__gm__ half*)key, k_total_);
    q_weight_.SetGlobalBuffer((__gm__ half*)q_weight, head_dim_);
    k_weight_.SetGlobalBuffer((__gm__ half*)k_weight, head_dim_);
    cos_.SetGlobalBuffer((__gm__ half*)cos, half_dim_);
    sin_.SetGlobalBuffer((__gm__ half*)sin, half_dim_);
    pipe_.InitBuffer(calc_buf_, kMaxHeadDim * sizeof(float) * 6);
  }

  __aicore__ inline void Process() {
    for (int64_t head = 0; head < q_heads_; ++head) {
      NormalizeAndRope(query_, q_weight_, head * head_dim_);
    }
    for (int64_t head = 0; head < kv_heads_; ++head) {
      NormalizeAndRope(key_, k_weight_, head * head_dim_);
    }
  }

 private:
  __aicore__ inline void NormalizeAndRope(GlobalTensor<half>& tensor, GlobalTensor<half>& weight, int64_t offset) {
    LocalTensor<float> values = calc_buf_.Get<float>();
    LocalTensor<float> weights = values[kMaxHeadDim];
    LocalTensor<float> square = values[kMaxHeadDim * 2];
    LocalTensor<float> reduce = values[kMaxHeadDim * 3];
    LocalTensor<float> cos_values = values[kMaxHeadDim * 4];
    LocalTensor<float> sin_values = values[kMaxHeadDim * 5];

    float square_sum = 0.0F;
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      const float value = static_cast<float>(tensor.GetValue(offset + dim));
      values.SetValue(dim, value);
      weights.SetValue(dim, static_cast<float>(weight.GetValue(dim)));
      square_sum += value * value;
    }
    const float scale = 1.0F / sqrt(square_sum / static_cast<float>(head_dim_) + 1e-5F);
    for (int64_t dim = 0; dim < head_dim_; ++dim) {
      values.SetValue(dim, values.GetValue(dim) * scale * weights.GetValue(dim));
    }
    for (int64_t dim = 0; dim < half_dim_; ++dim) {
      cos_values.SetValue(dim, static_cast<float>(cos_.GetValue(dim)));
      sin_values.SetValue(dim, static_cast<float>(sin_.GetValue(dim)));
    }
    for (int64_t dim = 0; dim < half_dim_; ++dim) {
      const float first = values.GetValue(dim);
      const float second = values.GetValue(dim + half_dim_);
      const float c = cos_values.GetValue(dim);
      const float s = sin_values.GetValue(dim);
      tensor.SetValue(offset + dim, static_cast<half>(first * c - second * s));
      tensor.SetValue(offset + half_dim_ + dim, static_cast<half>(second * c + first * s));
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> calc_buf_;
  GlobalTensor<half> query_;
  GlobalTensor<half> key_;
  GlobalTensor<half> q_weight_;
  GlobalTensor<half> k_weight_;
  GlobalTensor<half> cos_;
  GlobalTensor<half> sin_;
  int64_t q_heads_ = 0;
  int64_t kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t half_dim_ = 0;
  int64_t q_total_ = 0;
  int64_t k_total_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_qk_norm_rope(GM_ADDR query,
                                                              GM_ADDR key,
                                                              GM_ADDR q_weight,
                                                              GM_ADDR k_weight,
                                                              GM_ADDR cos,
                                                              GM_ADDR sin,
                                                              GM_ADDR workspace,
                                                              GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindQkNormRopeTiling* tiling_data = reinterpret_cast<__gm__ MiniMindQkNormRopeTiling*>(tiling);
  KernelMiniMindQkNormRope op;
  op.Init(query, key, q_weight, k_weight, cos, sin, tiling_data);
  op.Process();
}
