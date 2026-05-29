#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindVectorTiling {
  uint32_t total_length;
  uint32_t tile_length;
};

constexpr int32_t kBufferNum = 2;

class KernelMiniMindAdd {
 public:
  __aicore__ inline KernelMiniMindAdd() = default;

  __aicore__ inline void Init(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR out, const __gm__ MiniMindVectorTiling* tiling) {
    total_length_ = tiling->total_length;
    tile_length_ = tiling->tile_length;
    lhs_.SetGlobalBuffer((__gm__ half*)lhs, total_length_);
    rhs_.SetGlobalBuffer((__gm__ half*)rhs, total_length_);
    out_.SetGlobalBuffer((__gm__ half*)out, total_length_);
    pipe_.InitBuffer(lhs_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(rhs_queue_, kBufferNum, tile_length_ * sizeof(half));
    pipe_.InitBuffer(out_queue_, kBufferNum, tile_length_ * sizeof(half));
  }

  __aicore__ inline void Process() {
    for (int64_t offset = 0; offset < total_length_; offset += tile_length_) {
      const int64_t remaining = total_length_ - offset;
      const int64_t length = tile_length_ < remaining ? tile_length_ : remaining;
      CopyIn(offset, length);
      Compute(length);
      CopyOut(offset, length);
    }
  }

 private:
  __aicore__ inline void CopyIn(int64_t offset, int64_t length) {
    LocalTensor<half> lhs_local = lhs_queue_.AllocTensor<half>();
    LocalTensor<half> rhs_local = rhs_queue_.AllocTensor<half>();
    DataCopy(lhs_local, lhs_[offset], length);
    DataCopy(rhs_local, rhs_[offset], length);
    lhs_queue_.EnQue(lhs_local);
    rhs_queue_.EnQue(rhs_local);
  }

  __aicore__ inline void Compute(int64_t length) {
    LocalTensor<half> lhs_local = lhs_queue_.DeQue<half>();
    LocalTensor<half> rhs_local = rhs_queue_.DeQue<half>();
    LocalTensor<half> out_local = out_queue_.AllocTensor<half>();
    Add(out_local, lhs_local, rhs_local, length);
    out_queue_.EnQue(out_local);
    lhs_queue_.FreeTensor(lhs_local);
    rhs_queue_.FreeTensor(rhs_local);
  }

  __aicore__ inline void CopyOut(int64_t offset, int64_t length) {
    LocalTensor<half> out_local = out_queue_.DeQue<half>();
    DataCopy(out_[offset], out_local, length);
    out_queue_.FreeTensor(out_local);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, kBufferNum> lhs_queue_;
  TQue<TPosition::VECIN, kBufferNum> rhs_queue_;
  TQue<TPosition::VECOUT, kBufferNum> out_queue_;
  GlobalTensor<half> lhs_;
  GlobalTensor<half> rhs_;
  GlobalTensor<half> out_;
  int64_t total_length_ = 0;
  int64_t tile_length_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_add(GM_ADDR lhs,
                                                     GM_ADDR rhs,
                                                     GM_ADDR out,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindVectorTiling* tiling_data = reinterpret_cast<__gm__ MiniMindVectorTiling*>(tiling);
  KernelMiniMindAdd op;
  op.Init(lhs, rhs, out, tiling_data);
  op.Process();
}
