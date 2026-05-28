#include "minimind_orangepi/weights.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace minimind {

namespace {

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << shape[i];
  }
  out << ']';
  return out.str();
}

}  // namespace

void WeightManifest::add(std::string name, std::vector<int64_t> shape) {
  if (contains(name)) {
    throw std::invalid_argument("duplicate tensor in weight manifest: " + name);
  }
  tensors_.push_back(TensorSpec{std::move(name), std::move(shape), true});
}

bool WeightManifest::contains(const std::string& name) const {
  return std::any_of(tensors_.begin(), tensors_.end(), [&](const TensorSpec& tensor) {
    return tensor.name == name;
  });
}

const std::vector<int64_t>& WeightManifest::shape(const std::string& name) const {
  const auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const TensorSpec& tensor) {
    return tensor.name == name;
  });
  if (it == tensors_.end()) {
    throw std::out_of_range("tensor not found in weight manifest: " + name);
  }
  return it->shape;
}

std::vector<std::string> WeightManifest::names() const {
  std::vector<std::string> result;
  result.reserve(tensors_.size());
  for (const auto& tensor : tensors_) {
    result.push_back(tensor.name);
  }
  return result;
}

std::vector<std::string> validate_weight_manifest(
    const WeightManifest& manifest,
    const std::vector<TensorSpec>& expected) {
  std::vector<std::string> errors;
  for (const auto& spec : expected) {
    if (!manifest.contains(spec.name)) {
      if (spec.required) {
        errors.push_back("missing tensor: " + spec.name);
      }
      continue;
    }
    const auto& actual = manifest.shape(spec.name);
    if (actual != spec.shape) {
      errors.push_back("shape mismatch for " + spec.name + ": expected " +
                       shape_string(spec.shape) + ", got " + shape_string(actual));
    }
  }
  return errors;
}

}  // namespace minimind
