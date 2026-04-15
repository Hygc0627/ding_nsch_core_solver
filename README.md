# Ding NSCH Core Solver

这个仓库只保留当前 CHNS 主求解器的核心部分，便于单独发布、复用和继续做数值实验。

使用说明见：

- [`docs/solver_user_guide.md`](docs/solver_user_guide.md)

范围：

- 二维 MAC 交错网格 CHNS 求解器
- 模块化后的 `CH -> momentum predictor -> pressure correction` 主流程
- CH 隐式线性子问题的稀疏矩阵 / Krylov 求解
- 一个最小 smoke case
- 一组核心单元测试

目录：

- `src/core/base/`
  `Config`、均匀网格、场变量和诊断结构等基础模块。
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
- `testcases/05_static_droplet_dr1.cfg`、`06_static_droplet_dr10.cfg`、`07_static_droplet_dr100.cfg`
  用于回归验证的不同密度比静止液滴算例。

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/ding_nsch testcases/04_coupled_capillary_smoke.cfg
```

Restart 示例：

```bash
./build/ding_nsch testcases/13_restart_smoke_initial.cfg
./build/ding_nsch testcases/14_restart_smoke_resume.cfg
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
  真正的 ICC(0)-preconditioned CG。
- `ildlt_pcg`
  旧的 ILDLT-preconditioned CG 路径，已从原先误称的 `icpcg` 正名。
- `liu_split_icpcg` / `split_icpcg` / `paper_split_icpcg`
  将变系数压力泊松分裂成常系数隐式部分和显式变系数部分，对应
  `1/rho grad p = 1/rho_ref grad p + (1/rho - 1/rho_ref) grad(2p^l - p^{l-1})`，
  隐式常系数泊松用 ICC(0)-preconditioned CG。
- `liu_split_ildlt_pcg` / `split_ildlt_pcg` / `paper_split_ildlt_pcg`
  同一套 split Poisson 公式，但隐式常系数泊松用 ILDLT-preconditioned CG。
- `petsc_pcg`
  将压力矩阵/RHS 导出给 `python/petsc_pressure_solver.py`，由 `petsc4py` 调用 PETSc KSP 求解，参数在 `python/petsc_pressure_options.py` 或用户自定义 Python 配置中指定。

`petsc_pcg` 现在已经随仓库打包，但你仍然需要：

- 可用的 `petsc4py`
- 正确的 `PETSC_DIR` / `PETSC_ARCH`
- 在配置里指定合适的 `petsc_python_executable`，如果系统 `python3` 里没有 `petsc4py`

`hydea` 仍然保留为外部接口，需要你另外提供对应脚本和模型文件。

## Boundary Conditions

周期性仍然按方向配置：

- `periodic_x = true/false`
- `periodic_y = true/false`

对于非周期方向，现在可以分别给 `u`、`v`、`pressure` 的四条边设置：

- `*_bc_left_type`, `*_bc_left_value`
- `*_bc_right_type`, `*_bc_right_value`
- `*_bc_bottom_type`, `*_bc_bottom_value`
- `*_bc_top_type`, `*_bc_top_value`

其中 `*` 可替换成 `u`、`v` 或 `pressure`，类型支持：

- `dirichlet`
- `neumann`
- `unset`

更完整的说明见 [docs/solver_user_guide.md](docs/solver_user_guide.md)。
