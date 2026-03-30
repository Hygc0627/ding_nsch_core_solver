# Ding NSCH Core Solver

这个仓库只保留当前 CHNS 主求解器的核心部分，便于单独发布、复用和继续做数值实验。

范围：

- 二维 MAC 交错网格 CHNS 求解器
- 模块化后的 `CH -> momentum predictor -> pressure correction` 主流程
- CH 隐式线性子问题的稀疏矩阵 / Krylov 求解
- 一个最小 smoke case
- 一组核心单元测试

目录：

- `src/core/coupled/`
  主求解器实现，已经按算子、相场、动量、压力、诊断、I/O 分文件。
- `src/core/linear_algebra/`
  CH 线性系统使用的稀疏矩阵和 Krylov 组件。
- `src/main.cpp`
  命令行入口。
- `src/tests/core_solver_unit_tests.cpp`
  针对核心模块的最小回归测试。
- `testcases/04_coupled_capillary_smoke.cfg`
  最小耦合 smoke case。

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/ding_nsch testcases/04_coupled_capillary_smoke.cfg
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

或者直接运行：

```bash
./build/core_solver_unit_tests
```

## Pressure Backends

默认可直接使用的内置压力求解方式：

- `jacobi`
- `icpcg`
- `liu_split_icpcg`

代码中仍保留了 `petsc_pcg` 和 `hydea` 两个外部后端接口，但这个独立仓库没有打包对应 Python 驱动脚本。只有在你另外提供这些脚本，并正确设置 `PETSC_DIR` / `PETSC_ARCH` 环境变量时，才应启用它们。
