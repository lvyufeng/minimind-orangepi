#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindVectorTiling {
  int64_t total_length;
  int64_t tile_length;
};

constexpr int32_t kBufferNum = 2;

class KernelMiniMindSwiGlu {
 public:
  __aicore__ inline KernelMiniMindSwiGlu() = default;

  __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR y, const MiniMindVectorTiling* tiling) {
    total_length_ = tiling->total_length;
    tile_length_ = tiling->tile_length;
    block_length_ = (total_length_ + GetBlockNum() - 1) / GetBlockNum();
    const int64_t offset = GetBlockIdx() * block_length_;
    if (offset + block_length_ > total_length_) {
      block_length_ = total_length_ - offset;
    }
    gate_.SetGlobalBuffer((__gm__ half*)gate + offset, block_length_);
    up_.SetGlobalBuffer((__gm__ half*)up + offset, block_length_);
    y_.SetGlobalBuffer((__gm__ half*)y + offset, block_length_);
    pipe_.InitBuffer(gate_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(up_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(tmp_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(y_queue_, kBufferNum, tile_length_ * sizeof(half));
  }

  __aicore__ inline void Process() {
    for (int64_t offset = 0; offset < block_length_; offset += tile_length_) {
      const int64_t length = Min(tile_length_, block_length_ - offset);
      CopyIn(offset, length);
      Compute(length);
      CopyOut(offset, length);
    }
  }

 private:
  __aicore__ inline void CopyIn(int64_t offset, int64_t length) {
    LocalTensor<half> gate_local = gate_queue_.AllocTensor<half>();
    LocalTensor<half> up_local = up_queue_.AllocTensor<half>();
    DataCopy(gate_local, gate_[offset], length);
    DataCopy(up_local, up_[offset], length);
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
  GlobalTensor<half> gate_;
  GlobalTensor<half> up_;
  GlobalTensor<half> y_;
  int64_t total_length_ = 0;
  int64_t block_length_ = 0;
  int64_t tile_length_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_swi_glu(GM_ADDR gate,
                                                         GM_ADDR up,
                                                         GM_ADDR y,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  (void)workspace;
  const MiniMindVectorTiling* tiling_data = reinterpret_cast<const MiniMindVectorTiling*>(tiling);
  KernelMiniMindSwiGlu op;
  op.Init(gate, up, y, tiling_data);
  op.Process();
}
