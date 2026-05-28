#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindRopeTiling {
  uint32_t total_length;
  uint32_t metadata;
};

constexpr int32_t kBufferNum = 2;
constexpr int32_t kTilePairs = 128;

class KernelMiniMindRope {
 public:
  __aicore__ inline KernelMiniMindRope() = default;

  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR cos,
                              GM_ADDR sin,
                              GM_ADDR y,
                              const __gm__ MiniMindRopeTiling* tiling) {
    total_length_ = tiling->total_length;
    half_dim_ = tiling->metadata;
    head_dim_ = half_dim_ * 2;
    heads_ = total_length_ / head_dim_;
    x_.SetGlobalBuffer((__gm__ half*)x, total_length_);
    cos_.SetGlobalBuffer((__gm__ half*)cos, half_dim_);
    sin_.SetGlobalBuffer((__gm__ half*)sin, half_dim_);
    y_.SetGlobalBuffer((__gm__ half*)y, total_length_);
    pipe_.InitBuffer(first_queue_, kBufferNum, kTilePairs * sizeof(half));
    pipe_.InitBuffer(second_queue_, kBufferNum, kTilePairs * sizeof(half));
    pipe_.InitBuffer(cos_queue_, kBufferNum, kTilePairs * sizeof(half));
    pipe_.InitBuffer(sin_queue_, kBufferNum, kTilePairs * sizeof(half));
    pipe_.InitBuffer(out_first_queue_, kBufferNum, kTilePairs * sizeof(half));
    pipe_.InitBuffer(out_second_queue_, kBufferNum, kTilePairs * sizeof(half));
    pipe_.InitBuffer(calc_buf_, kTilePairs * sizeof(float) * 8);
  }

  __aicore__ inline void Process() {
    for (int64_t head = 0; head < heads_; ++head) {
      const int64_t head_offset = head * head_dim_;
      for (int64_t dim = 0; dim < half_dim_; dim += kTilePairs) {
        const int64_t remaining = half_dim_ - dim;
        const int64_t length = kTilePairs < remaining ? kTilePairs : remaining;
        CopyIn(head_offset, dim, length);
        Compute(length);
        CopyOut(head_offset, dim, length);
      }
    }
  }

 private:
  __aicore__ inline void CopyIn(int64_t head_offset, int64_t dim, int64_t length) {
    LocalTensor<half> first_local = first_queue_.AllocTensor<half>();
    LocalTensor<half> second_local = second_queue_.AllocTensor<half>();
    LocalTensor<half> cos_local = cos_queue_.AllocTensor<half>();
    LocalTensor<half> sin_local = sin_queue_.AllocTensor<half>();
    DataCopy(first_local, x_[head_offset + dim], length);
    DataCopy(second_local, x_[head_offset + half_dim_ + dim], length);
    DataCopy(cos_local, cos_[dim], length);
    DataCopy(sin_local, sin_[dim], length);
    first_queue_.EnQue(first_local);
    second_queue_.EnQue(second_local);
    cos_queue_.EnQue(cos_local);
    sin_queue_.EnQue(sin_local);
  }

  __aicore__ inline void Compute(int64_t length) {
    LocalTensor<half> first_local = first_queue_.DeQue<half>();
    LocalTensor<half> second_local = second_queue_.DeQue<half>();
    LocalTensor<half> cos_local = cos_queue_.DeQue<half>();
    LocalTensor<half> sin_local = sin_queue_.DeQue<half>();
    LocalTensor<half> out_first_local = out_first_queue_.AllocTensor<half>();
    LocalTensor<half> out_second_local = out_second_queue_.AllocTensor<half>();
    LocalTensor<float> first_float = calc_buf_.Get<float>();
    LocalTensor<float> second_float = first_float[kTilePairs];
    LocalTensor<float> cos_float = first_float[kTilePairs * 2];
    LocalTensor<float> sin_float = first_float[kTilePairs * 3];
    LocalTensor<float> tmp = first_float[kTilePairs * 4];
    LocalTensor<float> out_first = first_float[kTilePairs * 5];
    LocalTensor<float> out_second = first_float[kTilePairs * 6];

    Cast(first_float, first_local, RoundMode::CAST_NONE, length);
    Cast(second_float, second_local, RoundMode::CAST_NONE, length);
    Cast(cos_float, cos_local, RoundMode::CAST_NONE, length);
    Cast(sin_float, sin_local, RoundMode::CAST_NONE, length);
    PipeBarrier<PIPE_V>();
    Mul(out_first, first_float, cos_float, length);
    Mul(tmp, second_float, sin_float, length);
    PipeBarrier<PIPE_V>();
    Sub(out_first, out_first, tmp, length);
    PipeBarrier<PIPE_V>();
    Mul(out_second, second_float, cos_float, length);
    Mul(tmp, first_float, sin_float, length);
    PipeBarrier<PIPE_V>();
    Add(out_second, out_second, tmp, length);
    PipeBarrier<PIPE_V>();
    Cast(out_first_local, out_first, RoundMode::CAST_NONE, length);
    Cast(out_second_local, out_second, RoundMode::CAST_NONE, length);

    out_first_queue_.EnQue(out_first_local);
    out_second_queue_.EnQue(out_second_local);
    first_queue_.FreeTensor(first_local);
    second_queue_.FreeTensor(second_local);
    cos_queue_.FreeTensor(cos_local);
    sin_queue_.FreeTensor(sin_local);
  }

  __aicore__ inline void CopyOut(int64_t head_offset, int64_t dim, int64_t length) {
    LocalTensor<half> out_first_local = out_first_queue_.DeQue<half>();
    LocalTensor<half> out_second_local = out_second_queue_.DeQue<half>();
    DataCopy(y_[head_offset + dim], out_first_local, length);
    DataCopy(y_[head_offset + half_dim_ + dim], out_second_local, length);
    out_first_queue_.FreeTensor(out_first_local);
    out_second_queue_.FreeTensor(out_second_local);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, kBufferNum> first_queue_;
  TQue<TPosition::VECIN, kBufferNum> second_queue_;
  TQue<TPosition::VECIN, kBufferNum> cos_queue_;
  TQue<TPosition::VECIN, kBufferNum> sin_queue_;
  TQue<TPosition::VECOUT, kBufferNum> out_first_queue_;
  TQue<TPosition::VECOUT, kBufferNum> out_second_queue_;
  TBuf<TPosition::VECCALC> calc_buf_;
  GlobalTensor<half> x_;
  GlobalTensor<half> cos_;
  GlobalTensor<half> sin_;
  GlobalTensor<half> y_;
  int64_t total_length_ = 0;
  int64_t half_dim_ = 0;
  int64_t head_dim_ = 0;
  int64_t heads_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_rope(GM_ADDR x,
                                                      GM_ADDR cos,
                                                      GM_ADDR sin,
                                                      GM_ADDR y,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindRopeTiling* tiling_data = reinterpret_cast<__gm__ MiniMindRopeTiling*>(tiling);
  KernelMiniMindRope op;
  op.Init(x, cos, sin, y, tiling_data);
  op.Process();
}
