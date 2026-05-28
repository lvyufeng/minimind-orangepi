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

struct MiniMindRopeTiling {
  uint32_t total_length;
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

uint32_t infer_dtype_like_input0(gert::InferDataTypeContext* context) {
  return context->SetOutputDataType(0, context->GetInputDataType(0));
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

OP_ADD(MiniMindRmsNorm);
OP_ADD(MiniMindSwiGlu);
OP_ADD(MiniMindRope);

}  // namespace
