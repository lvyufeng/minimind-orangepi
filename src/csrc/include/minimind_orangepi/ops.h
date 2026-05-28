#pragma once

#include "minimind_orangepi/tensor.h"

namespace minimind {

HostTensor add(const HostTensor& lhs, const HostTensor& rhs);
HostTensor zeros(TensorShape shape, DType dtype);

}  // namespace minimind
