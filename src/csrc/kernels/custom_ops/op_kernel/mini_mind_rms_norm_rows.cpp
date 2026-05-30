#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindRmsNormRowsTiling {
  uint32_t rows;
  uint32_t cols;
};

class KernelMiniMindRmsNormRows {
 public:
  __aicore__ inline KernelMiniMindRmsNormRows() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y, const __gm__ MiniMindRmsNormRowsTiling* tiling) {
    rows_ = tiling->rows;
    cols_ = tiling->cols;
    cols_float_ = static_cast<float>(cols_);
    eps_ = 1e-5F;
    x_.SetGlobalBuffer((__gm__ half*)x, static_cast<int64_t>(rows_) * cols_);
    weight_.SetGlobalBuffer((__gm__ half*)weight, cols_);
    y_.SetGlobalBuffer((__gm__ half*)y, static_cast<int64_t>(rows_) * cols_);
    pipe_.InitBuffer(x_queue_, 1, cols_ * sizeof(half));
    pipe_.InitBuffer(w_queue_, 1, cols_ * sizeof(half));
    pipe_.InitBuffer(y_queue_, 1, cols_ * sizeof(half));
    pipe_.InitBuffer(calc_buf_, cols_ * sizeof(float) * 4);
  }

  __aicore__ inline void Process() {
    for (int64_t row = 0; row < rows_; ++row) {
      const int64_t offset = row * cols_;
      CopyInput(offset);
      const float square_sum = ComputeSquareSum();
      const float scale = 1.0F / sqrt(square_sum / cols_float_ + eps_);
      CopyInputAndWeight(offset);
      ComputeNorm(scale);
      CopyOut(offset);
    }
  }

 private:
  __aicore__ inline void CopyInput(int64_t offset) {
    LocalTensor<half> x_local = x_queue_.AllocTensor<half>();
    DataCopy(x_local, x_[offset], cols_);
    x_queue_.EnQue(x_local);
  }

  __aicore__ inline void CopyInputAndWeight(int64_t offset) {
    LocalTensor<half> x_local = x_queue_.AllocTensor<half>();
    LocalTensor<half> w_local = w_queue_.AllocTensor<half>();
    DataCopy(x_local, x_[offset], cols_);
    DataCopy(w_local, weight_, cols_);
    x_queue_.EnQue(x_local);
    w_queue_.EnQue(w_local);
  }

  __aicore__ inline float ComputeSquareSum() {
    LocalTensor<half> x_local = x_queue_.DeQue<half>();
    LocalTensor<float> work = calc_buf_.Get<float>();
    LocalTensor<float> square = work[cols_];
    LocalTensor<float> reduce = work[cols_ * 2];
    Cast(work, x_local, RoundMode::CAST_NONE, cols_);
    PipeBarrier<PIPE_V>();
    Mul(square, work, work, cols_);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduce, square, work, static_cast<int32_t>(cols_));
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float sum = reduce.GetValue(0);
    x_queue_.FreeTensor(x_local);
    return sum;
  }

  __aicore__ inline void ComputeNorm(float scale) {
    LocalTensor<half> x_local = x_queue_.DeQue<half>();
    LocalTensor<half> w_local = w_queue_.DeQue<half>();
    LocalTensor<half> y_local = y_queue_.AllocTensor<half>();
    LocalTensor<float> x_float = calc_buf_.Get<float>();
    LocalTensor<float> w_float = x_float[cols_];
    LocalTensor<float> y_float = x_float[cols_ * 2];

    Cast(x_float, x_local, RoundMode::CAST_NONE, cols_);
    Cast(w_float, w_local, RoundMode::CAST_NONE, cols_);
    PipeBarrier<PIPE_V>();
    Muls(y_float, x_float, scale, cols_);
    PipeBarrier<PIPE_V>();
    Mul(y_float, y_float, w_float, cols_);
    PipeBarrier<PIPE_V>();
    Cast(y_local, y_float, RoundMode::CAST_NONE, cols_);

    y_queue_.EnQue(y_local);
    x_queue_.FreeTensor(x_local);
    w_queue_.FreeTensor(w_local);
  }

  __aicore__ inline void CopyOut(int64_t offset) {
    LocalTensor<half> y_local = y_queue_.DeQue<half>();
    DataCopy(y_[offset], y_local, cols_);
    y_queue_.FreeTensor(y_local);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 1> x_queue_;
  TQue<TPosition::VECIN, 1> w_queue_;
  TQue<TPosition::VECOUT, 1> y_queue_;
  TBuf<TPosition::VECCALC> calc_buf_;
  GlobalTensor<half> x_;
  GlobalTensor<half> weight_;
  GlobalTensor<half> y_;
  int64_t rows_ = 0;
  int64_t cols_ = 0;
  float cols_float_ = 0.0F;
  float eps_ = 1e-5F;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_rms_norm_rows(GM_ADDR x,
                                                               GM_ADDR weight,
                                                               GM_ADDR y,
                                                               GM_ADDR workspace,
                                                               GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindRmsNormRowsTiling* tiling_data = reinterpret_cast<__gm__ MiniMindRmsNormRowsTiling*>(tiling);
  KernelMiniMindRmsNormRows op;
  op.Init(x, weight, y, tiling_data);
  op.Process();
}
