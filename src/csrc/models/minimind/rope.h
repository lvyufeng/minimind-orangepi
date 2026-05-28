#pragma once

#include "config.h"
#include "minimind_orangepi/tensor.h"

#include <cstdint>

namespace minimind::model {

struct RopeTables {
  HostTensor cos;
  HostTensor sin;
};

RopeTables make_rope_tables(const MiniMindConfig& config, int64_t positions);

}  // namespace minimind::model
