cmake_minimum_required(VERSION 3.16)
project(ding_nsch_core_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_library(core_coupled
  src/core/base/config.cpp
  src/core/coupled/solver.cpp
  src/core/coupled/operators.cpp
  src/core/coupled/phase_field.cpp
  src/core/coupled/momentum.cpp
  src/core/coupled/pressure.cpp
  src/core/coupled/pressure_analysis.cpp
  src/core/coupled/diagnostics.cpp
  src/core/coupled/io.cpp
)
target_include_directories(core_coupled PRIVATE src)
target_compile_options(core_coupled PRIVATE -O2 -Wall -Wextra -Wpedantic)

add_executable(ding_nsch
  src/main.cpp
)
target_include_directories(ding_nsch PRIVATE src)
target_link_libraries(ding_nsch PRIVATE core_coupled)
target_compile_options(ding_nsch PRIVATE -O2 -Wall -Wextra -Wpedantic)

add_executable(core_solver_unit_tests
  src/tests/core_solver_unit_tests.cpp
)
target_include_directories(core_solver_unit_tests PRIVATE src)
target_link_libraries(core_solver_unit_tests PRIVATE core_coupled)
target_compile_options(core_solver_unit_tests PRIVATE -O2 -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_test(NAME core_solver_unit_tests COMMAND core_solver_unit_tests)
endif()
