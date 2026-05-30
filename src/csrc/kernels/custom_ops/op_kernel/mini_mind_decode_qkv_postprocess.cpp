#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindDecodeQkvPostprocessTiling {
  uint32_t tokens;
  uint32_t metadata;
};

constexpr int32_t kMaxHeadDim = 128;
constexpr int32_t kBufferNum = 2;
constexpr int32_t kCopyTileLength = 256;

class KernelMiniMindDecodeQkvPostprocess {
 public:
  __aicore__ inline KernelMiniMindDecodeQkvPostprocess() = default;

  __aicore__ inline void Init(GM_ADDR qkv,
                              GM_ADDR q_weight,
                              GM_ADDR k_weight,
                              GM_ADDR cos,
                              GM_ADDR sin,
                              GM_ADDR keys,
                              GM_ADDR values,
                              GM_ADDR query,
                              const __gm__ MiniMindDecodeQkvPostprocessTiling* tiling) {
    tokens_ = tiling->tokens;
    head_dim_ = tiling->metadata & 0xFFFFU;
    kv_heads_ = (tiling->metadata >> 16U) & 0xFFU;
    q_heads_ = (tiling->metadata >> 24U) & 0xFFU;
    half_dim_ = head_dim_ / 2;
    q_size_ = q_heads_ * head_dim_;
    kv_size_ = kv_heads_ * head_dim_;
    qkv_.SetGlobalBuffer((__gm__ half*)qkv, q_size_ + kv_size_ * 2);
    q_weight_.SetGlobalBuffer((__gm__ half*)q_weight, head_dim_);
    k_weight_.SetGlobalBuffer((__gm__ half*)k_weight, head_dim_);
    cos_.SetGlobalBuffer((__gm__ half*)cos, half_dim_);
    sin_.SetGlobalBuffer((__gm__ half*)sin, half_dim_);
    keys_.SetGlobalBuffer((__gm__ half*)keys, tokens_ * kv_size_);
    values_.SetGlobalBuffer((__gm__ half*)values, tokens_ * kv_size_);
    query_.SetGlobalBuffer((__gm__ half*)query, q_size_);
    pipe_.InitBuffer(x_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(w_queue_, 1, kMaxHeadDim * sizeof(half));
    pipe_.InitBuffer(cos_queue_, 1, kMaxHeadDim / 2 * sizeof(half));
    pipe_.InitBuffer(sin_queue_, 1, kMaxHeadDim / 2 * sizeof(half));
    pipe_.InitBuffer(out_first_queue_, 1, kMaxHeadDim / 2 * sizeof(half));
    pipe_.InitBuffer(out_second_queue_, 1, kMaxHeadDim / 2 * sizeof(half));
    pipe_.InitBuffer(copy_queue_, kBufferNum, kCopyTileLength * sizeof(half));
    pipe_.InitBuffer(calc_buf_, kMaxHeadDim * sizeof(float) * 8);
  }

  __aicore__ inline void Process() {
    const int64_t block = GetBlockIdx();
    if (block < q_heads_) {
      ProcessNormRope(qkv_, block * head_dim_, q_weight_, query_, block * head_dim_);
    }
    if (block < kv_heads_) {
      const int64_t cache_offset = (tokens_ - 1) * kv_size_ + block * head_dim_;
      ProcessNormRope(qkv_, q_size_ + block * head_dim_, k_weight_, keys_, cache_offset);
      CopyValueToCache(q_size_ + kv_size_ + block * head_dim_, cache_offset);
    }
  }

 private:
  __aicore__ inline void ProcessNormRope(GlobalTensor<half>& input,
                                         int64_t input_offset,
                                         GlobalTensor<half>& weight,
                                         GlobalTensor<half>& output,
                                         int64_t output_offset) {
    CopyNormInputs(input, input_offset, weight);
    const float square_sum = ComputeSquareSum();
    const float scale = 1.0F / sqrt(square_sum / static_cast<float>(head_dim_) + 1e-5F);
    CopyNormInputs(input, input_offset, weight);
    ComputeNormRope(scale);
    CopyNormRopeOut(output, output_offset);
  }

  __aicore__ inline void CopyNormInputs(GlobalTensor<half>& input, int64_t input_offset, GlobalTensor<half>& weight) {
    LocalTensor<half> x_local = x_queue_.AllocTensor<half>();
    LocalTensor<half> w_local = w_queue_.AllocTensor<half>();
    DataCopy(x_local, input[input_offset], head_dim_);
    DataCopy(w_local, weight, head_dim_);
    x_queue_.EnQue(x_local);
    w_queue_.EnQue(w_local);
  }

  __aicore__ inline float ComputeSquareSum() {
    LocalTensor<half> x_local = x_queue_.DeQue<half>();
    LocalTensor<half> w_local = w_queue_.DeQue<half>();
    LocalTensor<float> work = calc_buf_.Get<float>();
    LocalTensor<float> square = work[kMaxHeadDim];
    LocalTensor<float> reduce = work[kMaxHeadDim * 2];
    Cast(work, x_local, RoundMode::CAST_NONE, head_dim_);
    PipeBarrier<PIPE_V>();
    Mul(square, work, work, head_dim_);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduce, square, work, static_cast<int32_t>(head_dim_));
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float sum = reduce.GetValue(0);
    x_queue_.FreeTensor(x_local);
    w_queue_.FreeTensor(w_local);
    return sum;
  }

  __aicore__ inline void ComputeNormRope(float scale) {
    LocalTensor<half> x_local = x_queue_.DeQue<half>();
    LocalTensor<half> w_local = w_queue_.DeQue<half>();
    LocalTensor<half> cos_local = cos_queue_.AllocTensor<half>();
    LocalTensor<half> sin_local = sin_queue_.AllocTensor<half>();
    LocalTensor<half> out_first_local = out_first_queue_.AllocTensor<half>();
    LocalTensor<half> out_second_local = out_second_queue_.AllocTensor<half>();
    DataCopy(cos_local, cos_, half_dim_);
    DataCopy(sin_local, sin_, half_dim_);
    LocalTensor<float> x_float = calc_buf_.Get<float>();
    LocalTensor<float> w_float = x_float[kMaxHeadDim];
    LocalTensor<float> y_float = x_float[kMaxHeadDim * 2];
    LocalTensor<float> cos_float = x_float[kMaxHeadDim * 3];
    LocalTensor<float> sin_float = x_float[kMaxHeadDim * 4];
    LocalTensor<float> tmp = x_float[kMaxHeadDim * 5];
    LocalTensor<float> out_first = x_float[kMaxHeadDim * 6];
    LocalTensor<float> out_second = x_float[kMaxHeadDim * 7];

    Cast(x_float, x_local, RoundMode::CAST_NONE, head_dim_);
    Cast(w_float, w_local, RoundMode::CAST_NONE, head_dim_);
    Cast(cos_float, cos_local, RoundMode::CAST_NONE, half_dim_);
    Cast(sin_float, sin_local, RoundMode::CAST_NONE, half_dim_);
    PipeBarrier<PIPE_V>();
    Muls(y_float, x_float, scale, head_dim_);
    PipeBarrier<PIPE_V>();
    Mul(y_float, y_float, w_float, head_dim_);
    PipeBarrier<PIPE_V>();

    Mul(out_first, y_float, cos_float, half_dim_);
    Mul(tmp, y_float[half_dim_], sin_float, half_dim_);
    PipeBarrier<PIPE_V>();
    Sub(out_first, out_first, tmp, half_dim_);
    PipeBarrier<PIPE_V>();
    Mul(out_second, y_float[half_dim_], cos_float, half_dim_);
    Mul(tmp, y_float, sin_float, half_dim_);
    PipeBarrier<PIPE_V>();
    Add(out_second, out_second, tmp, half_dim_);
    PipeBarrier<PIPE_V>();
    Cast(out_first_local, out_first, RoundMode::CAST_NONE, half_dim_);
    Cast(out_second_local, out_second, RoundMode::CAST_NONE, half_dim_);

    out_first_queue_.EnQue(out_first_local);
    out_second_queue_.EnQue(out_second_local);
    x_queue_.FreeTensor(x_local);
    w_queue_.FreeTensor(w_local);
    cos_queue_.FreeTensor(cos_local);
    sin_queue_.FreeTensor(sin_local);
  }

  __aicore__ inline void CopyNormRopeOut(GlobalTensor<half>& output, int64_t output_offset) {
    LocalTensor<half> out_first_local = out_first_queue_.DeQue<half>();
    LocalTensor<half> out_second_local = out_second_queue_.DeQue<half>();
    DataCopy(output[output_offset], out_first_local, half_dim_);
    DataCopy(output[output_offset + half_dim_], out_second_local, half_dim_);
    out_first_queue_.FreeTensor(out_first_local);
    out_second_queue_.FreeTensor(out_second_local);
  }

  __aicore__ inline void CopyValueToCache(int64_t value_offset, int64_t cache_offset) {
    for (int64_t offset = 0; offset < head_dim_; offset += kCopyTileLength) {
      const int64_t remaining = head_dim_ - offset;
      const int64_t copy_length = kCopyTileLength < remaining ? kCopyTileLength : remaining;
      LocalTensor<half> local = copy_queue_.AllocTensor<half>();
      DataCopy(local, qkv_[value_offset + offset], copy_length);
      copy_queue_.EnQue(local);
      LocalTensor<half> ready = copy_queue_.DeQue<half>();
      DataCopy(values_[cache_offset + offset], ready, copy_length);
      copy_queue_.FreeTensor(ready);
    }
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 1> x_queue_;
  TQue<TPosition::VECIN, 1> w_queue_;
  TQue<TPosition::VECIN, 1> cos_queue_;
  TQue<TPosition::VECIN, 1> sin_queue_;
  TQue<TPosition::VECOUT, 1> out_first_queue_;
  TQue<TPosition::VECOUT, 1> out_second_queue_;
  TQue<TPosition::VECIN, kBufferNum> copy_queue_;
  TBuf<TPosition::VECCALC> calc_buf_;
  GlobalTensor<half> qkv_;
  GlobalTensor<half> q_weight_;
  GlobalTensor<half> k_weight_;
  GlobalTensor<half> cos_;
  GlobalTensor<half> sin_;
  GlobalTensor<half> keys_;
  GlobalTensor<half> values_;
  GlobalTensor<half> query_;
  int64_t tokens_ = 0;
  int64_t q_heads_ = 0;
  int64_t kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t half_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_decode_qkv_postprocess(GM_ADDR qkv,
                                                                        GM_ADDR q_weight,
                                                                        GM_ADDR k_weight,
                                                                        GM_ADDR cos,
                                                                        GM_ADDR sin,
                                                                        GM_ADDR keys,
                                                                        GM_ADDR values,
                                                                        GM_ADDR query,
                                                                        GM_ADDR workspace,
                                                                        GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindDecodeQkvPostprocessTiling* tiling_data =
      reinterpret_cast<__gm__ MiniMindDecodeQkvPostprocessTiling*>(tiling);
  KernelMiniMindDecodeQkvPostprocess op;
  op.Init(qkv, q_weight, k_weight, cos, sin, keys, values, query, tiling_data);
  op.Process();
}
