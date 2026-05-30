#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint32_t kSuccess = ge::GRAPH_SUCCESS;
constexpr uint32_t kDefaultTileLength = 256;
struct MiniMindVectorTiling {
  uint32_t total_length;
  uint32_t tile_length;
};

struct MiniMindRmsNormRowsTiling {
  uint32_t rows;
  uint32_t cols;
};

struct MiniMindSwiGluRowsTiling {
  uint32_t rows;
  uint32_t cols;
};

struct MiniMindQkNormRopeTiling {
  uint32_t metadata;
  uint32_t position;
};

struct MiniMindQkvSplitRowsTiling {
  uint32_t rows;
  uint32_t metadata;
};

struct MiniMindRopeTiling {
  uint32_t total_length;
  uint32_t metadata;
};

struct MiniMindAttentionTiling {
  uint32_t tokens;
  uint32_t metadata;
};

struct MiniMindDecodeQkvPostprocessTiling {
  uint32_t tokens;
  uint32_t metadata;
};

struct MiniMindPrefillAttentionTiling {
  uint32_t seq_len;
  uint32_t metadata;
};

uint32_t infer_like_input0(gert::InferShapeContext* context) {
  const gert::Shape* input = context->GetInputShape(0);
  gert::Shape* output = context->GetOutputShape(0);
  if (input == nullptr || output == nullptr) {
    return ge::GRAPH_FAILED;
  }
  *output = *input;
  return kSuccess;
}

uint32_t infer_qkv_split_rows(gert::InferShapeContext* context) {
  const gert::Shape* input = context->GetInputShape(0);
  gert::Shape* query = context->GetOutputShape(0);
  gert::Shape* key = context->GetOutputShape(1);
  gert::Shape* value = context->GetOutputShape(2);
  if (input == nullptr || query == nullptr || key == nullptr || value == nullptr || input->GetDimNum() != 2 ||
      input->GetDim(0) <= 0 || input->GetDim(1) <= 0) {
    return ge::GRAPH_FAILED;
  }
  const auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const int64_t* q_size = attrs->GetInt(0);
  const int64_t* kv_size = attrs->GetInt(1);
  if (q_size == nullptr || kv_size == nullptr || *q_size <= 0 || *kv_size <= 0 || input->GetDim(1) != *q_size + 2 * *kv_size) {
    return ge::GRAPH_FAILED;
  }
  query->SetDimNum(2);
  query->SetDim(0, input->GetDim(0));
  query->SetDim(1, *q_size);
  key->SetDimNum(2);
  key->SetDim(0, input->GetDim(0));
  key->SetDim(1, *kv_size);
  value->SetDimNum(2);
  value->SetDim(0, input->GetDim(0));
  value->SetDim(1, *kv_size);
  return kSuccess;
}

uint32_t infer_decode_qkv_postprocess(gert::InferShapeContext* context) {
  const gert::Shape* qkv = context->GetInputShape(0);
  gert::Shape* query = context->GetOutputShape(0);
  if (qkv == nullptr || query == nullptr || qkv->GetDimNum() != 2 || qkv->GetDim(0) != 1 || qkv->GetDim(1) <= 0) {
    return ge::GRAPH_FAILED;
  }
  const auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const int64_t* q_heads = attrs->GetInt(0);
  const int64_t* kv_heads = attrs->GetInt(1);
  const int64_t* head_dim = attrs->GetInt(2);
  if (q_heads == nullptr || kv_heads == nullptr || head_dim == nullptr || *q_heads <= 0 || *kv_heads <= 0 ||
      *head_dim <= 0 || qkv->GetDim(1) != (*q_heads + 2 * *kv_heads) * *head_dim) {
    return ge::GRAPH_FAILED;
  }
  query->SetDimNum(2);
  query->SetDim(0, *q_heads);
  query->SetDim(1, *head_dim);
  return kSuccess;
}

uint32_t infer_swiglu_rows(gert::InferShapeContext* context) {
  const gert::Shape* input = context->GetInputShape(0);
  gert::Shape* output = context->GetOutputShape(0);
  if (input == nullptr || output == nullptr || input->GetDimNum() != 2 || input->GetDim(1) <= 0 ||
      input->GetDim(1) % 2 != 0) {
    return ge::GRAPH_FAILED;
  }
  output->SetDimNum(2);
  output->SetDim(0, input->GetDim(0));
  output->SetDim(1, input->GetDim(1) / 2);
  return kSuccess;
}

uint32_t infer_prefill_attention(gert::InferShapeContext* context) {
  const gert::Shape* query = context->GetInputShape(0);
  gert::Shape* output = context->GetOutputShape(0);
  if (query == nullptr || output == nullptr) {
    return ge::GRAPH_FAILED;
  }
  *output = *query;
  return kSuccess;
}

uint32_t infer_dtype_like_input0(gert::InferDataTypeContext* context) {
  return context->SetOutputDataType(0, context->GetInputDataType(0));
}

uint32_t infer_dtype_qkv_split(gert::InferDataTypeContext* context) {
  const ge::DataType dtype = context->GetInputDataType(0);
  if (context->SetOutputDataType(0, dtype) != ge::GRAPH_SUCCESS ||
      context->SetOutputDataType(1, dtype) != ge::GRAPH_SUCCESS ||
      context->SetOutputDataType(2, dtype) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

int64_t shape_num_elements(const gert::Shape& shape) {
  int64_t size = 1;
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    size *= shape.GetDim(i);
  }
  return size;
}

uint32_t tile_vector_op(gert::TilingContext* context) {
  const gert::StorageShape* input = context->GetInputShape(0);
  if (input == nullptr) {
    return ge::GRAPH_FAILED;
  }

  MiniMindVectorTiling* tiling = context->GetTilingData<MiniMindVectorTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const int64_t total_length = shape_num_elements(input->GetStorageShape());
  const uint32_t blocks = 1;
  tiling->total_length = static_cast<uint32_t>(total_length);
  tiling->tile_length = kDefaultTileLength;

  if (context->SetBlockDim(blocks) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_rope_op(gert::TilingContext* context) {
  const gert::StorageShape* input = context->GetInputShape(0);
  if (input == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& shape = input->GetStorageShape();
  if (shape.GetDimNum() != 2 || shape.GetDim(1) <= 0 || shape.GetDim(1) % 2 != 0) {
    return ge::GRAPH_FAILED;
  }

  MiniMindRopeTiling* tiling = context->GetTilingData<MiniMindRopeTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }

  tiling->total_length = static_cast<uint32_t>(shape_num_elements(shape));
  tiling->metadata = static_cast<uint32_t>(shape.GetDim(1) / 2);

  if (context->SetBlockDim(1) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_attention_op(gert::TilingContext* context) {
  const gert::StorageShape* query = context->GetInputShape(0);
  const gert::StorageShape* keys = context->GetInputShape(1);
  if (query == nullptr || keys == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& query_shape = query->GetStorageShape();
  const gert::Shape& key_shape = keys->GetStorageShape();
  if (query_shape.GetDimNum() != 2 || key_shape.GetDimNum() != 3 || query_shape.GetDim(0) <= 0 ||
      query_shape.GetDim(0) > 255 || query_shape.GetDim(1) <= 0 || query_shape.GetDim(1) > 128 ||
      key_shape.GetDim(0) <= 0 || key_shape.GetDim(0) > 2048 || key_shape.GetDim(1) <= 0 ||
      key_shape.GetDim(1) > 255 || query_shape.GetDim(1) != key_shape.GetDim(2)) {
    return ge::GRAPH_FAILED;
  }

  MiniMindAttentionTiling* tiling = context->GetTilingData<MiniMindAttentionTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }

  tiling->tokens = static_cast<uint32_t>(key_shape.GetDim(0));
  tiling->metadata = (static_cast<uint32_t>(query_shape.GetDim(0)) << 24U) |
                     (static_cast<uint32_t>(key_shape.GetDim(1)) << 16U) |
                     static_cast<uint32_t>(query_shape.GetDim(1));

  const uint32_t blocks = static_cast<uint32_t>(query_shape.GetDim(0));
  if (context->SetBlockDim(blocks) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_decode_attention_op(gert::TilingContext* context) {
  const gert::StorageShape* query = context->GetInputShape(0);
  const gert::StorageShape* key = context->GetInputShape(1);
  const gert::StorageShape* value = context->GetInputShape(2);
  const gert::StorageShape* keys = context->GetInputShape(3);
  const gert::StorageShape* values = context->GetInputShape(4);
  if (query == nullptr || key == nullptr || value == nullptr || keys == nullptr || values == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& query_shape = query->GetStorageShape();
  const gert::Shape& key_shape = key->GetStorageShape();
  const gert::Shape& value_shape = value->GetStorageShape();
  const gert::Shape& keys_shape = keys->GetStorageShape();
  const gert::Shape& values_shape = values->GetStorageShape();
  if (query_shape.GetDimNum() != 2 || key_shape.GetDimNum() != 2 || value_shape.GetDimNum() != 2 ||
      keys_shape.GetDimNum() != 3 || values_shape.GetDimNum() != 3 || query_shape.GetDim(0) <= 0 ||
      query_shape.GetDim(0) > 255 || query_shape.GetDim(1) <= 0 || query_shape.GetDim(1) > 128 ||
      key_shape.GetDim(0) <= 0 || key_shape.GetDim(0) > 255 || key_shape.GetDim(1) != query_shape.GetDim(1) ||
      value_shape.GetDim(0) != key_shape.GetDim(0) || value_shape.GetDim(1) != key_shape.GetDim(1) ||
      keys_shape.GetDim(0) <= 0 || keys_shape.GetDim(0) > 2048 || keys_shape.GetDim(1) != key_shape.GetDim(0) ||
      keys_shape.GetDim(2) != query_shape.GetDim(1) || values_shape.GetDim(0) != keys_shape.GetDim(0) ||
      values_shape.GetDim(1) != keys_shape.GetDim(1) || values_shape.GetDim(2) != keys_shape.GetDim(2)) {
    return ge::GRAPH_FAILED;
  }

  MiniMindAttentionTiling* tiling = context->GetTilingData<MiniMindAttentionTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }

  tiling->tokens = static_cast<uint32_t>(keys_shape.GetDim(0));
  tiling->metadata = (static_cast<uint32_t>(query_shape.GetDim(0)) << 24U) |
                     (static_cast<uint32_t>(key_shape.GetDim(0)) << 16U) |
                     static_cast<uint32_t>(query_shape.GetDim(1));

  const uint32_t blocks = static_cast<uint32_t>(query_shape.GetDim(0));
  if (context->SetBlockDim(blocks) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_prefill_attention_op(gert::TilingContext* context) {
  const gert::StorageShape* query = context->GetInputShape(0);
  const gert::StorageShape* keys = context->GetInputShape(1);
  if (query == nullptr || keys == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& query_shape = query->GetStorageShape();
  const gert::Shape& key_shape = keys->GetStorageShape();
  if (query_shape.GetDimNum() != 3 || key_shape.GetDimNum() != 3 || query_shape.GetDim(0) <= 0 ||
      query_shape.GetDim(0) > 2048 || query_shape.GetDim(1) <= 0 || query_shape.GetDim(1) > 255 ||
      query_shape.GetDim(2) <= 0 || query_shape.GetDim(2) > 128 || key_shape.GetDim(0) != query_shape.GetDim(0) ||
      key_shape.GetDim(1) <= 0 || key_shape.GetDim(1) > 255 || key_shape.GetDim(2) != query_shape.GetDim(2)) {
    return ge::GRAPH_FAILED;
  }

  MiniMindPrefillAttentionTiling* tiling = context->GetTilingData<MiniMindPrefillAttentionTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }

  tiling->seq_len = static_cast<uint32_t>(query_shape.GetDim(0));
  tiling->metadata = (static_cast<uint32_t>(query_shape.GetDim(1)) << 24U) |
                     (static_cast<uint32_t>(key_shape.GetDim(1)) << 16U) |
                     static_cast<uint32_t>(query_shape.GetDim(2));

  if (context->SetBlockDim(1) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}


uint32_t tile_rms_norm_rows_op(gert::TilingContext* context) {
  const gert::StorageShape* input = context->GetInputShape(0);
  const gert::StorageShape* weight = context->GetInputShape(1);
  if (input == nullptr || weight == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& shape = input->GetStorageShape();
  const gert::Shape& weight_shape = weight->GetStorageShape();
  if (shape.GetDimNum() != 2 || weight_shape.GetDimNum() != 1 || shape.GetDim(0) <= 0 || shape.GetDim(1) <= 0 ||
      weight_shape.GetDim(0) != shape.GetDim(1)) {
    return ge::GRAPH_FAILED;
  }

  MiniMindRmsNormRowsTiling* tiling = context->GetTilingData<MiniMindRmsNormRowsTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }

  tiling->rows = static_cast<uint32_t>(shape.GetDim(0));
  tiling->cols = static_cast<uint32_t>(shape.GetDim(1));

  if (context->SetBlockDim(1) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_qk_norm_rope_op(gert::TilingContext* context) {
  const gert::StorageShape* query = context->GetInputShape(0);
  const gert::StorageShape* key = context->GetInputShape(1);
  if (query == nullptr || key == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape& query_shape = query->GetStorageShape();
  const gert::Shape& key_shape = key->GetStorageShape();
  if (query_shape.GetDimNum() != 2 || key_shape.GetDimNum() != 2 || query_shape.GetDim(0) <= 0 ||
      query_shape.GetDim(0) > 255 || key_shape.GetDim(0) <= 0 || key_shape.GetDim(0) > 255 ||
      query_shape.GetDim(1) <= 0 || query_shape.GetDim(1) > 128 || key_shape.GetDim(1) != query_shape.GetDim(1)) {
    return ge::GRAPH_FAILED;
  }
  MiniMindQkNormRopeTiling* tiling = context->GetTilingData<MiniMindQkNormRopeTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling->metadata = (static_cast<uint32_t>(query_shape.GetDim(0)) << 16U) |
                     (static_cast<uint32_t>(key_shape.GetDim(0)) << 8U) |
                     static_cast<uint32_t>(query_shape.GetDim(1));
  tiling->position = 0;
  if (context->SetBlockDim(1) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_qkv_split_rows_op(gert::TilingContext* context) {
  const gert::StorageShape* input = context->GetInputShape(0);
  if (input == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape& shape = input->GetStorageShape();
  if (shape.GetDimNum() != 2 || shape.GetDim(0) <= 0 || shape.GetDim(1) <= 0) {
    return ge::GRAPH_FAILED;
  }
  const auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const int64_t* q_size = attrs->GetInt(0);
  const int64_t* kv_size = attrs->GetInt(1);
  if (q_size == nullptr || kv_size == nullptr || *q_size <= 0 || *q_size > 65535 || *kv_size <= 0 ||
      *kv_size > 65535 || shape.GetDim(1) != *q_size + 2 * *kv_size) {
    return ge::GRAPH_FAILED;
  }
  MiniMindQkvSplitRowsTiling* tiling = context->GetTilingData<MiniMindQkvSplitRowsTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling->rows = static_cast<uint32_t>(shape.GetDim(0));
  tiling->metadata = (static_cast<uint32_t>(*q_size) << 16U) | static_cast<uint32_t>(*kv_size);
  if (context->SetBlockDim(1) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_decode_qkv_postprocess_op(gert::TilingContext* context) {
  const gert::StorageShape* qkv = context->GetInputShape(0);
  const gert::StorageShape* keys = context->GetInputShape(5);
  if (qkv == nullptr || keys == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape& qkv_shape = qkv->GetStorageShape();
  const gert::Shape& keys_shape = keys->GetStorageShape();
  if (qkv_shape.GetDimNum() != 2 || qkv_shape.GetDim(0) != 1 || keys_shape.GetDimNum() != 3 ||
      keys_shape.GetDim(0) <= 0 || keys_shape.GetDim(0) > 2048 || keys_shape.GetDim(1) <= 0 ||
      keys_shape.GetDim(1) > 255 || keys_shape.GetDim(2) <= 0 || keys_shape.GetDim(2) > 128) {
    return ge::GRAPH_FAILED;
  }
  const auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const int64_t* q_heads = attrs->GetInt(0);
  const int64_t* kv_heads = attrs->GetInt(1);
  const int64_t* head_dim = attrs->GetInt(2);
  const int64_t* tokens = attrs->GetInt(3);
  if (q_heads == nullptr || kv_heads == nullptr || head_dim == nullptr || tokens == nullptr || *q_heads <= 0 ||
      *q_heads > 255 || *kv_heads <= 0 || *kv_heads > 255 || *head_dim <= 0 || *head_dim > 128 || *tokens <= 0 ||
      *tokens > keys_shape.GetDim(0) || keys_shape.GetDim(1) != *kv_heads || keys_shape.GetDim(2) != *head_dim ||
      qkv_shape.GetDim(1) != (*q_heads + 2 * *kv_heads) * *head_dim) {
    return ge::GRAPH_FAILED;
  }
  MiniMindDecodeQkvPostprocessTiling* tiling = context->GetTilingData<MiniMindDecodeQkvPostprocessTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling->tokens = static_cast<uint32_t>(*tokens);
  tiling->metadata = (static_cast<uint32_t>(*q_heads) << 24U) |
                     (static_cast<uint32_t>(*kv_heads) << 16U) |
                     static_cast<uint32_t>(*head_dim);
  const uint32_t blocks = static_cast<uint32_t>(std::max(*q_heads, *kv_heads));
  if (context->SetBlockDim(blocks) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

uint32_t tile_swiglu_rows_op(gert::TilingContext* context) {
  const gert::StorageShape* input = context->GetInputShape(0);
  if (input == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& shape = input->GetStorageShape();
  if (shape.GetDimNum() != 2 || shape.GetDim(0) <= 0 || shape.GetDim(1) <= 0 || shape.GetDim(1) % 2 != 0) {
    return ge::GRAPH_FAILED;
  }

  MiniMindSwiGluRowsTiling* tiling = context->GetTilingData<MiniMindSwiGluRowsTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }

  tiling->rows = static_cast<uint32_t>(shape.GetDim(0));
  tiling->cols = static_cast<uint32_t>(shape.GetDim(1) / 2);

  if (context->SetBlockDim(1) != ge::GRAPH_SUCCESS || context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return kSuccess;
}

class MiniMindRmsNorm : public ops::OpDef {
 public:
  explicit MiniMindRmsNorm(const char* name) : OpDef(name) {
    Input("x").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("weight").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("y").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Attr("eps").Float(1e-5F);
    SetInferShape(infer_like_input0);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_vector_op).AddConfig("ascend310b");
  }
};

class MiniMindRmsNormRows : public ops::OpDef {
 public:
  explicit MiniMindRmsNormRows(const char* name) : OpDef(name) {
    Input("x").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("weight").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("y").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Attr("eps").Float(1e-5F);
    SetInferShape(infer_like_input0);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_rms_norm_rows_op).AddConfig("ascend310b");
  }
};

class MiniMindSwiGlu : public ops::OpDef {
 public:
  explicit MiniMindSwiGlu(const char* name) : OpDef(name) {
    Input("gate").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("up").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("y").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_like_input0);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_vector_op).AddConfig("ascend310b");
  }
};

class MiniMindQkNormRope : public ops::OpDef {
 public:
  explicit MiniMindQkNormRope(const char* name) : OpDef(name) {
    Input("query").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("key").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("q_weight").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("k_weight").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("cos").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("sin").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_prefill_attention);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_qk_norm_rope_op).AddConfig("ascend310b");
  }
};

class MiniMindDecodeQkvPostprocess : public ops::OpDef {
 public:
  explicit MiniMindDecodeQkvPostprocess(const char* name) : OpDef(name) {
    Input("qkv").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("q_weight").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("k_weight").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("cos").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("sin").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("keys").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("values").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("query").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Attr("q_heads").Int(0);
    Attr("kv_heads").Int(0);
    Attr("head_dim").Int(0);
    Attr("tokens").Int(0);
    SetInferShape(infer_decode_qkv_postprocess);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_decode_qkv_postprocess_op).AddConfig("ascend310b");
  }
};

class MiniMindQkvSplitRows : public ops::OpDef {
 public:
  explicit MiniMindQkvSplitRows(const char* name) : OpDef(name) {
    Input("qkv").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("query").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("key").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("value").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Attr("q_size").Int(0);
    Attr("kv_size").Int(0);
    SetInferShape(infer_qkv_split_rows);
    SetInferDataType(infer_dtype_qkv_split);
    AICore().SetTiling(tile_qkv_split_rows_op).AddConfig("ascend310b");
  }
};

class MiniMindSwiGluRows : public ops::OpDef {
 public:
  explicit MiniMindSwiGluRows(const char* name) : OpDef(name) {
    Input("gate_up").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("y").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_swiglu_rows);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_swiglu_rows_op).AddConfig("ascend310b");
  }
};

class MiniMindRope : public ops::OpDef {
 public:
  explicit MiniMindRope(const char* name) : OpDef(name) {
    Input("x").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("cos").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("sin").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("y").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_like_input0);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_rope_op).AddConfig("ascend310b");
  }
};

class MiniMindAttention : public ops::OpDef {
 public:
  explicit MiniMindAttention(const char* name) : OpDef(name) {
    Input("query").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("keys").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("values").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("out").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_like_input0);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_attention_op).AddConfig("ascend310b");
  }
};

class MiniMindDecodeAttention : public ops::OpDef {
 public:
  explicit MiniMindDecodeAttention(const char* name) : OpDef(name) {
    Input("query").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("key").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("value").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("keys").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("values").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("out").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_like_input0);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_decode_attention_op).AddConfig("ascend310b");
  }
};

class MiniMindPrefillAttention : public ops::OpDef {
 public:
  explicit MiniMindPrefillAttention(const char* name) : OpDef(name) {
    Input("query").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("keys").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("values").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("out").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_prefill_attention);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_prefill_attention_op).AddConfig("ascend310b");
  }
};

class MiniMindAdd : public ops::OpDef {
 public:
  explicit MiniMindAdd(const char* name) : OpDef(name) {
    Input("lhs").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Input("rhs").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    Output("out").ParamType(ops::REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    SetInferShape(infer_like_input0);
    SetInferDataType(infer_dtype_like_input0);
    AICore().SetTiling(tile_vector_op).AddConfig("ascend310b");
  }
};

OP_ADD(MiniMindRmsNorm);
OP_ADD(MiniMindRmsNormRows);
OP_ADD(MiniMindSwiGlu);
OP_ADD(MiniMindQkNormRope);
OP_ADD(MiniMindDecodeQkvPostprocess);
OP_ADD(MiniMindQkvSplitRows);
OP_ADD(MiniMindSwiGluRows);
OP_ADD(MiniMindRope);
OP_ADD(MiniMindAttention);
OP_ADD(MiniMindDecodeAttention);
OP_ADD(MiniMindPrefillAttention);
OP_ADD(MiniMindAdd);

}  // namespace
