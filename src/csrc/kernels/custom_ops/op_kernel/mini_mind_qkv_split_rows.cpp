#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct MiniMindQkvSplitRowsTiling {
  uint32_t rows;
  uint32_t metadata;
};

constexpr int32_t kBufferNum = 2;
constexpr int64_t kTileLength = 256;

class KernelMiniMindQkvSplitRows {
 public:
  __aicore__ inline KernelMiniMindQkvSplitRows() = default;

  __aicore__ inline void Init(GM_ADDR qkv,
                              GM_ADDR query,
                              GM_ADDR key,
                              GM_ADDR value,
                              const __gm__ MiniMindQkvSplitRowsTiling* tiling) {
    rows_ = tiling->rows;
    q_size_ = (tiling->metadata >> 16U) & 0xFFFFU;
    kv_size_ = tiling->metadata & 0xFFFFU;
    row_size_ = q_size_ + kv_size_ * 2;
    qkv_.SetGlobalBuffer((__gm__ half*)qkv, rows_ * row_size_);
    query_.SetGlobalBuffer((__gm__ half*)query, rows_ * q_size_);
    key_.SetGlobalBuffer((__gm__ half*)key, rows_ * kv_size_);
    value_.SetGlobalBuffer((__gm__ half*)value, rows_ * kv_size_);
    pipe_.InitBuffer(copy_queue_, kBufferNum, kTileLength * sizeof(half));
  }

  __aicore__ inline void Process() {
    for (int64_t row = 0; row < rows_; ++row) {
      CopySegment(row * row_size_, row * q_size_, q_size_, query_);
      CopySegment(row * row_size_ + q_size_, row * kv_size_, kv_size_, key_);
      CopySegment(row * row_size_ + q_size_ + kv_size_, row * kv_size_, kv_size_, value_);
    }
  }

 private:
  __aicore__ inline void CopySegment(int64_t input_offset,
                                     int64_t output_offset,
                                     int64_t length,
                                     GlobalTensor<half>& output) {
    for (int64_t offset = 0; offset < length; offset += kTileLength) {
      const int64_t remaining = length - offset;
      const int64_t copy_length = kTileLength < remaining ? kTileLength : remaining;
      LocalTensor<half> local = copy_queue_.AllocTensor<half>();
      DataCopy(local, qkv_[input_offset + offset], copy_length);
      copy_queue_.EnQue(local);
      LocalTensor<half> ready = copy_queue_.DeQue<half>();
      DataCopy(output[output_offset + offset], ready, copy_length);
      copy_queue_.FreeTensor(ready);
    }
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, kBufferNum> copy_queue_;
  GlobalTensor<half> qkv_;
  GlobalTensor<half> query_;
  GlobalTensor<half> key_;
  GlobalTensor<half> value_;
  int64_t rows_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  int64_t row_size_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mini_mind_qkv_split_rows(GM_ADDR qkv,
                                                                GM_ADDR query,
                                                                GM_ADDR key,
                                                                GM_ADDR value,
                                                                GM_ADDR workspace,
                                                                GM_ADDR tiling) {
  (void)workspace;
  __gm__ MiniMindQkvSplitRowsTiling* tiling_data = reinterpret_cast<__gm__ MiniMindQkvSplitRowsTiling*>(tiling);
  KernelMiniMindQkvSplitRows op;
  op.Init(qkv, query, key, value, tiling_data);
  op.Process();
}
