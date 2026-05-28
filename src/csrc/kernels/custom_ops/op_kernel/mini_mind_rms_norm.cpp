#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindVectorTiling {
  int64_t total_length;
  int64_t tile_length;
  float eps;
};

constexpr int32_t kBufferNum = 2;

class KernelMiniMindRmsNorm {
 public:
  __aicore__ inline KernelMiniMindRmsNorm() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y, const MiniMindVectorTiling* tiling) {
    total_length_ = tiling->total_length;
    tile_length_ = tiling->tile_length;
    eps_ = tiling->eps;
    x_.SetGlobalBuffer((__gm__ half*)x, total_length_);
    weight_.SetGlobalBuffer((__gm__ half*)weight, total_length_);
    y_.SetGlobalBuffer((__gm__ half*)y, total_length_);
    pipe_.InitBuffer(x_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(w_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(y_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(calc_buf_, tile_length_ * sizeof(float) * 4);
  }

  __aicore__ inline void Process() {
    float square_sum = 0.0F;
    for (int64_t offset = 0; offset < total_length_; offset += tile_length_) {
      const int64_t length = Min(tile_length_, total_length_ - offset);
      CopyInput(offset, length);
      square_sum += ComputeSquareSum(length);
    }

    const float scale = 1.0F / sqrt(square_sum / static_cast<float>(total_length_) + eps_);
    for (int64_t offset = 0; offset < total_length_; offset += tile_length_) {
      const int64_t length = Min(tile_length_, total_length_ - offset);
      CopyInputAndWeight(offset, length);
      ComputeNorm(scale, length);
      CopyOut(offset, length);
    }
  }

 private:
  __aicore__ inline void CopyInput(int64_t offset, int64_t length) {
    LocalTensor<half> x_local = x_queue_.AllocTensor<half>();
    DataCopy(x_local, x_[offset], length);
    x_queue_.EnQue(x_local);
  }

  __aicore__ inline void CopyInputAndWeight(int64_t offset, int64_t length) {
    LocalTensor<half> x_local = x_queue_.AllocTensor<half>();
    LocalTensor<half> w_local = w_queue_.AllocTensor<half>();
    DataCopy(x_local, x_[offset], length);
    DataCopy(w_local, weight_[offset], length);
    x_queue_.EnQue(x_local);
    w_queue_.EnQue(w_local);
  }

  __aicore__ inline float ComputeSquareSum(int64_t length) {
    LocalTensor<half> x_local = x_queue_.DeQue<half>();
    LocalTensor<float> work = calc_buf_.Get<float>();
    LocalTensor<float> square = work[tile_length_];
    LocalTensor<float> reduce = work[tile_length_ * 2];
    Cast(work, x_local, RoundMode::CAST_NONE, length);
    PipeBarrier<PIPE_V>();
    Mul(square, work, work, length);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduce, square, work, static_cast<int32_t>(length));
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float sum = reduce.GetValue(0);
    x_queue_.FreeTensor(x_local);
    return sum;
  }

  __aicore__ inline void ComputeNorm(float scale, int64_t length) {
    LocalTensor<half> x_local = x_queue_.DeQue<half>();
    LocalTensor<half> w_local = w_queue_.DeQue<half>();
    LocalTensor<half> y_local = y_queue_.AllocTensor<half>();
    LocalTensor<float> x_float = calc_buf_.Get<float>();
    LocalTensor<float> w_float = x_float[tile_length_];
    LocalTensor<float> y_float = x_float[tile_length_ * 2];

    Cast(x_float, x_local, RoundMode::CAST_NONE, length);
    Cast(w_float, w_local, RoundMode::CAST_NONE, length);
    PipeBarrier<PIPE_V>();
    Muls(y_float, x_float, scale, length);
    PipeBarrier<PIPE_V>();
    Mul(y_float, y_float, w_float, length);
    PipeBarrier<PIPE_V>();
    Cast(y_local, y_float, RoundMode::CAST_NONE, length);

    y_queue_.EnQue(y_local);
    x_queue_.FreeTensor(x_local);
    w_queue_.FreeTensor(w_local);
  }

  __aicore__ inline void CopyOut(int64_t offset, int64_t length) {
    LocalTensor<half> y_local = y_queue_.DeQue<half>();
    DataCopy(y_[offset], y_local, length);
    y_queue_.FreeTensor(y_local);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, kBufferNum> x_queue_;
  TQue<TPosition::VECIN, kBufferNum> w_queue_;
  TQue<TPosition::VECOUT, kBufferNum> y_queue_;
  TBuf<TPosition::VECCALC> calc_buf_;
  GlobalTensor<half> x_;
  GlobalTensor<half> weight_;
  GlobalTensor<half> y_;
  int64_t total_length_ = 0;
  int64_t tile_length_ = 0;
  float eps_ = 1e-5F;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_rms_norm(GM_ADDR x,
                                                          GM_ADDR weight,
                                                          GM_ADDR y,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling) {
  (void)workspace;
  const MiniMindVectorTiling* tiling_data = reinterpret_cast<const MiniMindVectorTiling*>(tiling);
  KernelMiniMindRmsNorm op;
  op.Init(x, weight, y, tiling_data);
  op.Process();
}
