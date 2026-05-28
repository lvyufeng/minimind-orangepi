#pragma once

#include "minimind_orangepi/tensor.h"

namespace minimind {

HostTensor add(const HostTensor& lhs, const HostTensor& rhs);
HostTensor zeros(TensorShape shape, DType dtype);
HostTensor rms_norm(const HostTensor& input, const HostTensor& weight, float eps);
HostTensor swiglu(const HostTensor& gate, const HostTensor& up);

}  // namespace minimind
