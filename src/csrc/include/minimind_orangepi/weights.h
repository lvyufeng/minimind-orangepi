#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minimind {

struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  bool required = true;
};

class WeightManifest {
 public:
  void add(std::string name, std::vector<int64_t> shape);
  bool contains(const std::string& name) const;
  const std::vector<int64_t>& shape(const std::string& name) const;
  std::vector<std::string> names() const;

 private:
  std::vector<TensorSpec> tensors_;
};

std::vector<std::string> validate_weight_manifest(
    const WeightManifest& manifest,
    const std::vector<TensorSpec>& expected);

}  // namespace minimind
