#include "core/coupled/solver.hpp"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ding_nsch <case.cfg>\n";
    return 1;
  }

  try {
    ding::Config cfg = ding::load_config(argv[1]);
    ding::Solver solver(cfg);
    return solver.run() ? 0 : 2;
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
