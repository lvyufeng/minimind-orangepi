#pragma once

#include "language_model.h"

#include <string>

namespace minimind::model {

LanguageModel load_runtime_language_model(const std::string& model_dir);
bool has_runtime_language_model(const std::string& model_dir);

}  // namespace minimind::model
