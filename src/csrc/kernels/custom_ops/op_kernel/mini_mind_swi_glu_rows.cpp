#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindSwiGluRowsTiling {
  uint32_t rows;
  uint32_t cols;
};

constexpr int32_t kBufferNum = 2;
constexpr int64_t kTileLength = 256;

class KernelMiniMindSwiGluRows {
 public:
  __aicore__ inline KernelMiniMindSwiGluRows() = default;

  __aicore__ inline void Init(GM_ADDR gate_up, GM_ADDR y, const __gm__ MiniMindSwiGluRowsTiling* tiling) {
    rows_ = tiling->rows;
    cols_ = tiling->cols;
    tile_length_ = kTileLength;
    gate_up_.SetGlobalBuffer((__gm__ half*)gate_up, rows_ * cols_ * 2);
    y_.SetGlobalBuffer((__gm__ half*)y, rows_ * cols_);
    pipe_.InitBuffer(gate_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(up_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(tmp_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(y_queue_, kBufferNum, tile_length_ * sizeof(half));
  }

  __aicore__ inline void Process() {
    for (int64_t row = 0; row < rows_; ++row) {
      const int64_t gate_base = row * cols_ * 2;
      const int64_t up_base = gate_base + cols_;
      const int64_t out_base = row * cols_;
      for (int64_t col = 0; col < cols_; col += tile_length_) {
        const int64_t remaining = cols_ - col;
        const int64_t length = tile_length_ < remaining ? tile_length_ : remaining;
        CopyIn(gate_base + col, up_base + col, length);
        Compute(length);
        CopyOut(out_base + col, length);
      }
    }
  }

 private:
  __aicore__ inline void CopyIn(int64_t gate_offset, int64_t up_offset, int64_t length) {
    LocalTensor<half> gate_local = gate_queue_.AllocTensor<half>();
    LocalTensor<half> up_local = up_queue_.AllocTensor<half>();
    DataCopy(gate_local, gate_up_[gate_offset], length);
    DataCopy(up_local, gate_up_[up_offset], length);
    gate_queue_.EnQue(gate_local);
    up_queue_.EnQue(up_local);
  }

  __aicore__ inline void Compute(int64_t length) {
    LocalTensor<half> gate_local = gate_queue_.DeQue<half>();
    LocalTensor<half> up_local = up_queue_.DeQue<half>();
    LocalTensor<half> tmp_local = tmp_queue_.AllocTensor<half>();
    LocalTensor<half> y_local = y_queue_.AllocTensor<half>();

    Muls(tmp_local, gate_local, static_cast<half>(-1.0F), length);
    Exp(tmp_local, tmp_local, length);
    Adds(tmp_local, tmp_local, static_cast<half>(1.0F), length);
    Div(tmp_local, gate_local, tmp_local, length);
    Mul(y_local, tmp_local, up_local, length);

    y_queue_.EnQue(y_local);
    gate_queue_.FreeTensor(gate_local);
    up_queue_.FreeTensor(up_local);
    tmp_queue_.FreeTensor(tmp_local);
  }

  __aicore__ inline void CopyOut(int64_t offset, int64_t length) {
    LocalTensor<half> y_local = y_queue_.DeQue<half>();
    DataCopy(y_[offset], y_local, length);
    y_queue_.FreeTensor(y_local);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, kBufferNum> gate_queue_;
  TQue<TPosition::VECIN, kBufferNum> up_queue_;
  TQue<TPosition::VECIN, kBufferNum> tmp_queue_;
  TQue<TPosition::VECOUT, kBufferNum> y_queue_;
  GlobalTensor<half> gate_up_;
  GlobalTensor<half> y_;
  int64_t rows_ = 0;
  int64_t cols_ = 0;
  int64_t tile_length_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_swi_glu_rows(GM_ADDR gate_up,
                                                              GM_ADDR y,
                                                              GM_ADDR workspace,
                                                              GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindSwiGluRowsTiling* tiling_data = reinterpret_cast<__gm__ MiniMindSwiGluRowsTiling*>(tiling);
  KernelMiniMindSwiGluRows op;
  op.Init(gate_up, y, tiling_data);
  op.Process();
}
