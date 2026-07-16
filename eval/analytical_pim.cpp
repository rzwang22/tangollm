#include "analytical/analytical_pim.h"

#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  const std::string config_path =
      argc > 1 ? std::string(argv[1]) : std::string("analytical_pim_config.yaml");
  std::vector<std::string> overrides;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--set") {
      if (index + 1 >= argc) {
        return 2;
      }
      overrides.emplace_back(argv[++index]);
    } else if (argument.rfind("--set=", 0) == 0) {
      overrides.push_back(argument.substr(6));
    } else {
      return 2;
    }
  }
  return llm_system::analytical::RunAnalyticalPIM(config_path, overrides);
}
