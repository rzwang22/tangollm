#include "analytical/analytical_pim.h"

#include <string>

int main(int argc, char* argv[]) {
  const std::string config_path =
      argc > 1 ? std::string(argv[1]) : std::string("analytical_pim_config.yaml");
  return llm_system::analytical::RunAnalyticalPIM(config_path);
}
