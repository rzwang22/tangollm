#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm_system {
namespace analytical {

int RunAnalyticalPIM(const std::string& config_path,
                     const std::vector<std::string>& overrides = {});

}  // namespace analytical
}  // namespace llm_system
